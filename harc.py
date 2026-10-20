# -*- coding: utf-8 -*-
# Copyright (c) 2026 yuyu2929
#
# 双许可：
#   * 非商业使用（个人学习/科研/非营利组织）：PolyForm Noncommercial 1.0.0
#     见 LICENSE-POLYFORM.md
#   * 商业使用（企业/营利组织）：需购买商业许可，见 COMMERCIAL-LICENSE.md
"""
HARC —— Hybrid Adaptive Raster Coding（混合自适应栅格编码）v1.0
================================================================

设计目标：融合各类栅格编码之长，补齐各自短板（无损编码）。

  结构层  四叉树递归分解 —— 吸收四叉树编码：变分辨率、均质区域大块合并
  叶子层  节点级"三选一"择优：
            UNIFORM  整块同值      -> 只存一个增量值        （块编码思想）
            RLE      行内长游程    -> 游程对编码            （游程编码：简单、可叠置）
            PRED     破碎/复杂纹理 -> 中值边缘预测差分+zlib （PNG/LERC 思想，
                                              专治游程编码对破碎数据的"膨胀"）
  熵层    叶子载荷统一 zlib 压缩 —— 吸收现代通用压缩

补齐的传统缺点：
  * 游程编码对破碎数据膨胀      -> 自动回落到 PRED 熵编码叶子
  * 四叉树"改一动全"            -> 叶子独立编码，支持单叶重编（set_pixel）
  * 链式编码无法参与叠置分析    -> 全程保持栅格结构
  * 直接编码零压缩              -> 全链路压缩
  * 整图解压才能读一角          -> 区域随机解码 decode_region()

码流布局（所有整数 = zigzag + varint）：
  [magic 'HARC'][version][height][width][min_leaf]
  树（深度优先，NW-NE-SW-SE）：
    TAG=0 UNIFORM : varint(zigzag(value - ctx))
    TAG=1 RLE     : varint(len) + varint(对数) + 对*{varint(值增量), varint(游程长)}
    TAG=2 PRED    : varint(len) + zlib(中值预测残差)
    TAG=3 QUAD    : 4 个子节点（子块尺寸由 (h,w) 递推，无需存储）
"""

from __future__ import annotations
import zlib
import numpy as np

MAGIC = b"HARC"
__version__ = "1.1.0"      # 包版本（与码流格式版本 VERSION 无关）
VERSION = 2          # v2：头部增加 dtype 字节 + 尾部增加 CRC32；读端仍兼容 v1
VERSION_MIN = 1
TAG_UNIFORM, TAG_RLE, TAG_PRED, TAG_QUAD = 0, 1, 2, 3

# 支持的 dtype 及编码（v1 无 dtype 字段，一律按 int64 处理）
DTYPE_CODES = {
    np.dtype(np.bool_): 1,
    np.dtype(np.int8): 2,
    np.dtype(np.uint8): 3,
    np.dtype(np.int16): 4,
    np.dtype(np.uint16): 5,
    np.dtype(np.int32): 6,
    np.dtype(np.uint32): 7,
    np.dtype(np.int64): 8,
    np.dtype(np.uint64): 9,
}
CODE_DTYPES = {v: k for k, v in DTYPE_CODES.items()}
LEGACY_DTYPE_CODE = 8                     # v1 码流默认 int64
MAX_SIDE = 1 << 20                        # 单边上限 2^20，防止畸形头导致超大分配


# ---------------- varint / zigzag ----------------

def zigzag(v: int) -> int:
    return 2 * v if v >= 0 else -2 * v - 1


def unzigzag(z: int) -> int:
    return (z >> 1) if (z & 1) == 0 else -((z + 1) >> 1)


def put_varint(buf: bytearray, v: int) -> None:
    # 注意：这里必须是显式检查，不能写 assert —— python -O 会移除断言，
    # 而负值会让下面的 while True 永远无法收敛（v >>= 7 对负数保持负数）。
    if v < 0:
        raise ValueError(f"varint 不接受负数: {v}")
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            return


class Reader:
    __slots__ = ("data", "pos", "limit")

    def __init__(self, data, pos=0, limit=None):
        self.data = data
        self.pos = pos
        self.limit = len(data) if limit is None else limit

    def need(self, n: int = 1) -> None:
        if n < 0 or self.pos + n > self.limit:
            raise ValueError(f"码流截断：需要 {n} 字节，剩余 {max(0, self.limit - self.pos)}")

    def byte(self) -> int:
        self.need(1)
        b = self.data[self.pos]
        self.pos += 1
        return b

    def take(self, n: int) -> bytes:
        self.need(n)
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def varint(self) -> int:
        v, shift = 0, 0
        while True:
            b = self.byte()
            v |= (b & 0x7F) << shift
            if not (b & 0x80):
                return v
            shift += 7
            if shift > 63:
                raise ValueError("varint 过长（超过 64 位）")


# ---------------- 预测差分（编码向量化 / 解码逐行） ----------------

def _median_residual(block: np.ndarray) -> np.ndarray:
    """PNG 式中值边缘预测：p = clamp(tl, min(l,t), max(l,t))。"""
    b = block.astype(np.int64)
    l = np.zeros_like(b); l[:, 1:] = b[:, :-1]
    t = np.zeros_like(b); t[1:, :] = b[:-1, :]
    tl = np.zeros_like(b); tl[1:, 1:] = b[:-1, :-1]
    lo = np.minimum(l, t); hi = np.maximum(l, t)
    med = np.minimum(np.maximum(tl, lo), hi)
    return b - med


def _median_reconstruct(res: np.ndarray) -> np.ndarray:
    """逆预测：行内像素有顺序依赖，逐行重建。"""
    h, w = res.shape
    out = np.empty((h, w), dtype=np.int64)
    prev = np.zeros(w, dtype=np.int64)
    for i in range(h):
        r = res[i]
        rec = np.empty(w, dtype=np.int64)
        left = 0
        for j in range(w):
            up = prev[j]
            tl = prev[j - 1] if j > 0 else 0
            lo = left if left < up else up
            hi = left if left > up else up
            p = tl
            if p < lo:
                p = lo
            elif p > hi:
                p = hi
            left = int(r[j]) + p
            rec[j] = left
        out[i] = rec
        prev = rec
    return out


# ---------------- 叶子编码 ----------------

def _rle_pairs(block: np.ndarray, ctx: int):
    flat = block.reshape(-1)
    if flat.size == 1:
        return [(zigzag(int(flat[0]) - ctx), 1)]
    idx = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], idx))
    ends = np.concatenate((idx, [flat.size]))
    pairs, prev = [], ctx
    for s, e in zip(starts, ends):
        v = int(flat[s])
        pairs.append((zigzag(v - prev), int(e - s)))
        prev = v
    return pairs


def _enc_rle_bytes(block, ctx) -> bytes:
    pairs = _rle_pairs(block, ctx)
    buf = bytearray()
    put_varint(buf, len(pairs))
    for dz, run in pairs:
        put_varint(buf, dz)
        put_varint(buf, run)
    return bytes(buf)


def _dec_rle(payload: bytes, h: int, w: int, ctx: int) -> np.ndarray:
    rd = Reader(payload)
    npairs = rd.varint()
    flat = np.empty(h * w, dtype=np.int64)
    p, v = 0, ctx
    for _ in range(npairs):
        v += unzigzag(rd.varint())
        run = rd.varint()
        flat[p:p + run] = v
        p += run
    assert p == h * w, "RLE 游程与块尺寸不符"
    return flat.reshape(h, w)


# PRED 残差的动态位宽：按块内 zigzag 最大值选最窄无符号整型，
# 头部 1 字节存位宽码。平滑数据残差小，通常 uint8/uint16 即可，
# 相比固定 uint64 可省 4~8 倍存储。
_PRED_DT = ((np.uint8, 255), (np.uint16, 65535),
            (np.uint32, 2**32 - 1), (np.uint64, None))


def _enc_pred_bytes(block) -> bytes:
    res = _median_residual(block)
    # 在 uint64 空间做 zigzag：直接写 2*abs(res) 会让 INT64_MIN 取绝对值溢出
    # （abs(INT64_MIN) 仍然是 INT64_MIN），从而把极值静默截断成垃圾数据。
    r = res.view(np.uint64)
    z = r << np.uint64(1)
    neg = (r >> np.uint64(63)).astype(bool)          # 负数需异或全 1
    z = z.copy()
    z[neg] ^= np.uint64(0xFFFFFFFFFFFFFFFF)
    mx = int(z.max()) if z.size else 0
    for code, (dt, cap) in enumerate(_PRED_DT):
        if cap is None or mx <= cap:
            break
    return bytes([code]) + zlib.compress(z.astype(dt).tobytes(), 6)


def _dec_pred(payload: bytes, h: int, w: int) -> np.ndarray:
    if not payload:
        raise ValueError("PRED 载荷为空")
    code = payload[0]
    if code >= len(_PRED_DT):
        raise ValueError(f"未知 PRED 位宽码：{code}")
    dt = _PRED_DT[code][0]
    try:
        raw = zlib.decompress(payload[1:])
    except zlib.error as e:
        raise ValueError(f"PRED 载荷解压失败：{e}") from e
    z = np.frombuffer(raw, dtype=dt)
    if z.size != h * w:
        raise ValueError(f"PRED 残差数量不符（{z.size} != {h * w}）")
    zu = z.astype(np.uint64)
    bits = zu >> np.uint64(1)
    odd = (zu & np.uint64(1)).astype(bool)
    bits = bits.copy()
    bits[odd] ^= np.uint64(0xFFFFFFFFFFFFFFFF)
    res = bits.view(np.int64).reshape(h, w)
    return _median_reconstruct(res)


def _dec_leaf(tag: int, payload: bytes, h: int, w: int, ctx: int) -> np.ndarray:
    if tag == TAG_RLE:
        return _dec_rle(payload, h, w, ctx)
    return _dec_pred(payload, h, w)


def _best_leaf(block, ctx):
    """RLE 与 PRED 择优，返回 (tag, payload)。"""
    rle = _enc_rle_bytes(block, ctx)
    pred = _enc_pred_bytes(block)
    if len(rle) <= len(pred):
        return TAG_RLE, rle
    return TAG_PRED, pred


# ---------------- 主编码 / 全图解码 ----------------

def encode(arr: np.ndarray, min_leaf: int = 32) -> bytes:
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError("仅支持二维栅格")
    if a.size == 0:
        raise ValueError("空数组无法编码")
    if a.shape[0] > MAX_SIDE or a.shape[1] > MAX_SIDE:
        raise ValueError(f"单边尺寸超过上限 {MAX_SIDE}")
    if min_leaf < 1:
        raise ValueError("min_leaf 必须 >= 1")

    code = DTYPE_CODES.get(a.dtype)
    if code is None:
        raise TypeError(f"不支持的 dtype：{a.dtype}（支持 8/16/32/64 位整数与 bool）")
    if a.dtype == np.uint64 and a.size and int(a.max()) > np.iinfo(np.int64).max:
        raise ValueError("uint64 数值超出 int64 范围，暂不支持")

    head = bytearray()
    head += MAGIC
    put_varint(head, VERSION)
    put_varint(head, code)                             # v2：新增 dtype 字节
    put_varint(head, a.shape[0])
    put_varint(head, a.shape[1])
    put_varint(head, min_leaf)

    def enc_node(b: np.ndarray, ctx: int) -> bytes:
        h, w = b.shape
        if np.all(b == b[0, 0]):                       # UNIFORM：任何层级都最优先
            out = bytearray([TAG_UNIFORM])
            put_varint(out, zigzag(int(b[0, 0]) - ctx))
            return bytes(out)

        # 任一边不足 2 时无法四分（会产生 0 高/宽子块），强制成叶
        is_leaf = (h <= min_leaf and w <= min_leaf) or h < 2 or w < 2
        rle = _enc_rle_bytes(b, ctx)
        pred = _enc_pred_bytes(b)
        cand = [(len(rle), TAG_RLE, rle), (len(pred), TAG_PRED, pred)]

        if not is_leaf:                                # 内部节点允许"提前成叶"
            ch, cw = h // 2, w // 2
            sub = bytearray()
            for r0, r1, c0, c1 in ((0, ch, 0, cw), (0, ch, cw, w),
                                   (ch, h, 0, cw), (ch, h, cw, w)):
                sub += enc_node(b[r0:r1, c0:c1], ctx)
            cand.append((len(sub) + 1, TAG_QUAD, bytes(sub)))

        cand.sort(key=lambda x: x[0])
        _, tag, payload = cand[0]
        out = bytearray([tag])
        if tag in (TAG_RLE, TAG_PRED):
            put_varint(out, len(payload))
        out += payload                                 # RLE/PRED: 载荷；QUAD: 全部子节点
        return bytes(out)

    body = enc_node(a.astype(np.int64, copy=False), 0)
    return _finalize(bytes(head), body)


def _finalize(head: bytes, body: bytes) -> bytes:
    """v2：整段（头+体）算 CRC32 附在尾部。"""
    payload = head + body
    return payload + zlib.crc32(payload).to_bytes(4, "little")


def _read_header(data: bytes):
    """解析头部，返回 (reader, h, w, min_leaf, dtype_code, version)。

    v2 会顺带校验尾部 CRC32；reader.limit 已排除 CRC 那 4 字节。
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("码流必须是 bytes")
    data = bytes(data)
    if len(data) < 8 or data[:4] != MAGIC:
        raise ValueError("不是 HARC 码流（magic 不匹配或长度不足）")
    rd = Reader(data, 4)
    version = rd.varint()
    if not (VERSION_MIN <= version <= VERSION):
        raise ValueError(f"不支持的版本：{version}")
    code = rd.varint() if version >= 2 else LEGACY_DTYPE_CODE
    if code not in CODE_DTYPES:
        raise ValueError(f"未知 dtype 编码：{code}")
    h, w = rd.varint(), rd.varint()
    min_leaf = rd.varint()
    if not (0 < h <= MAX_SIDE and 0 < w <= MAX_SIDE):
        raise ValueError(f"尺寸非法：{h}x{w}")
    if min_leaf < 1:
        raise ValueError("min_leaf 非法")

    if version >= 2:
        if len(data) < rd.pos + 4:
            raise ValueError("码流截断：缺少 CRC32")
        body_end = len(data) - 4
        want = int.from_bytes(data[body_end:], "little")
        got = zlib.crc32(data[:body_end]) & 0xFFFFFFFF
        if want != got:
            raise ValueError(f"CRC32 校验失败（期望 {want:08x}，实际 {got:08x}），数据已损坏")
        rd.limit = body_end

    return rd, h, w, min_leaf, code, version


def _skip_node(rd: Reader, hh: int, ww: int) -> None:
    """跳过一个节点（不解码）：叶子跳载荷，QUAD 递归跳子节点。"""
    tag = rd.data[rd.pos]; rd.pos += 1
    if tag in (TAG_RLE, TAG_PRED):
        ln = rd.varint()                           # 注意：不能写 rd.pos += rd.varint()
        rd.pos += ln                               # 增强赋值会先取旧 pos，导致差 varint 字节数
    elif tag == TAG_UNIFORM:
        rd.varint()
    else:
        ch, cw = hh // 2, ww // 2
        for qr0, qr1, qc0, qc1 in ((0, ch, 0, cw), (0, ch, cw, ww),
                                   (ch, hh, 0, cw), (ch, hh, cw, ww)):
            _skip_node(rd, qr1 - qr0, qc1 - qc0)


def decode(data: bytes) -> np.ndarray:
    rd, h, w, _min_leaf, code, _ver = _read_header(data)

    def dec_node(hh: int, ww: int, ctx: int) -> np.ndarray:
        tag = rd.byte()
        if tag == TAG_UNIFORM:
            return np.full((hh, ww), ctx + unzigzag(rd.varint()), dtype=np.int64)
        if tag in (TAG_RLE, TAG_PRED):
            ln = rd.varint()
            payload = rd.take(ln)
            return _dec_leaf(tag, payload, hh, ww, ctx)
        ch, cw = hh // 2, ww // 2                      # QUAD
        out = np.empty((hh, ww), dtype=np.int64)
        out[:ch, :cw] = dec_node(ch, cw, ctx)
        out[:ch, cw:] = dec_node(ch, ww - cw, ctx)
        out[ch:, :cw] = dec_node(hh - ch, cw, ctx)
        out[ch:, cw:] = dec_node(hh - ch, ww - cw, ctx)
        return out

    out = dec_node(h, w, 0)
    if rd.pos != rd.limit:
        raise ValueError(f"码流尾部有多余数据（解析到 {rd.pos}，应为 {rd.limit}）")
    return out.astype(CODE_DTYPES[code], copy=False)


# ---------------- 区域随机解码 ----------------

def decode_region(data: bytes, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    rd, h, w, _min_leaf, code, _ver = _read_header(data)
    r0, r1 = max(0, r0), min(h, r1)
    c0, c1 = max(0, c0), min(w, c1)
    if r1 <= r0 or c1 <= c0:
        return np.empty((max(0, r1 - r0), max(0, c1 - c0)), dtype=CODE_DTYPES[code])
    out = np.empty((r1 - r0, c1 - c0), dtype=np.int64)

    def skip(hh: int, ww: int) -> None:
        tag = rd.byte()
        if tag in (TAG_RLE, TAG_PRED):
            ln = rd.varint()
            rd.pos += ln
            rd.need(0)
        elif tag == TAG_UNIFORM:
            rd.varint()
        else:
            ch, cw = hh // 2, ww // 2
            skip(ch, cw); skip(ch, ww - cw)
            skip(hh - ch, cw); skip(hh - ch, ww - cw)

    def fill(hh: int, ww: int, gr0: int, gc0: int, ctx: int) -> None:
        tag = rd.byte()
        ir0, ir1 = max(r0, gr0), min(r1, gr0 + hh)
        ic0, ic1 = max(c0, gc0), min(c1, gc0 + ww)
        hit = ir0 < ir1 and ic0 < ic1

        if tag == TAG_UNIFORM:
            v = ctx + unzigzag(rd.varint())
            if hit:
                out[ir0 - r0:ir1 - r0, ic0 - c0:ic1 - c0] = v
        elif tag in (TAG_RLE, TAG_PRED):
            ln = rd.varint()
            payload = rd.take(ln)
            if hit:
                blk = _dec_leaf(tag, payload, hh, ww, ctx)
                out[ir0 - r0:ir1 - r0, ic0 - c0:ic1 - c0] = \
                    blk[ir0 - gr0:ir1 - gr0, ic0 - gc0:ic1 - gc0]
        else:                                          # QUAD
            ch, cw = hh // 2, ww // 2
            for qr0, qr1, qc0, qc1 in ((0, ch, 0, cw), (0, ch, cw, ww),
                                       (ch, hh, 0, cw), (ch, hh, cw, ww)):
                sr0, sr1 = gr0 + qr0, gr0 + qr1
                sc0, sc1 = gc0 + qc0, gc0 + qc1
                if sr1 <= r0 or sr0 >= r1 or sc1 <= c0 or sc0 >= c1:
                    skip(qr1 - qr0, qc1 - qc0)         # 子块与窗口无交集：只跳过
                else:
                    fill(qr1 - qr0, qc1 - qc0, sr0, sc0, ctx)

    fill(h, w, 0, 0, 0)
    if rd.pos != rd.limit:
        raise ValueError(f"码流尾部有多余数据（解析到 {rd.pos}，应为 {rd.limit}）")
    return out.astype(CODE_DTYPES[code], copy=False)


# ---------------- 局部更新：只重编像素所在叶子 ----------------

def set_pixel(data: bytes, r: int, c: int, value: int) -> bytes:
    rd, h, w, min_leaf, code, _ver = _read_header(data)
    if not (0 <= r < h and 0 <= c < w):
        raise IndexError(f"坐标越界：({r},{c}) 不在 {h}x{w} 内")

    # --- 1) 定位叶子：记录其在码流中的起始偏移、几何信息与上下文 ---
    def walk(hh, ww, gr0, gc0, ctx):
        start = rd.pos
        tag = rd.data[rd.pos]; rd.pos += 1
        if tag == TAG_QUAD:
            ch, cw = hh // 2, ww // 2
            for qr0, qr1, qc0, qc1 in ((0, ch, 0, cw), (0, ch, cw, ww),
                                       (ch, hh, 0, cw), (ch, hh, cw, ww)):
                if gr0 + qr0 <= r < gr0 + qr1 and gc0 + qc0 <= c < gc0 + qc1:
                    return walk(qr1 - qr0, qc1 - qc0, gr0 + qr0, gc0 + qc0, ctx)
                _skip_node(rd, qr1 - qr0, qc1 - qc0)   # 兄弟子块必须跳过
            raise IndexError("坐标越界")
        if tag == TAG_UNIFORM:
            blk = np.full((hh, ww), ctx + unzigzag(rd.varint()), dtype=np.int64)
        else:
            ln = rd.varint()
            blk = _dec_leaf(tag, rd.data[rd.pos:rd.pos + ln], hh, ww, ctx)
            rd.pos += ln
        return start, blk, ctx

    leaf_start, blk, leaf_ctx = walk(h, w, 0, 0, 0)
    # 叶子绝对原点：从定位路径推不出来，用几何重走一遍得到
    # （叶子尺寸已知，配合 walk 无法直接给出原点，这里重走）
    rd2, *_ = _read_header(data)

    def find_origin(hh, ww, gr0, gc0):
        tag = rd2.data[rd2.pos]; rd2.pos += 1
        if tag == TAG_QUAD:
            ch, cw = hh // 2, ww // 2
            for qr0, qr1, qc0, qc1 in ((0, ch, 0, cw), (0, ch, cw, ww),
                                       (ch, hh, 0, cw), (ch, hh, cw, ww)):
                if gr0 + qr0 <= r < gr0 + qr1 and gc0 + qc0 <= c < gc0 + qc1:
                    return find_origin(qr1 - qr0, qc1 - qc0, gr0 + qr0, gc0 + qc0)
                _skip_node(rd2, qr1 - qr0, qc1 - qc0)
            raise IndexError
        if tag == TAG_UNIFORM:
            rd2.varint()
        else:
            ln2 = rd2.varint()
            rd2.pos += ln2
        return gr0, gc0

    lg_r0, lg_c0 = find_origin(h, w, 0, 0)
    blk[r - lg_r0, c - lg_c0] = value

    # --- 2) 择优重编该叶子 ---
    if np.all(blk == blk[0, 0]):
        new_tag, new_payload = TAG_UNIFORM, None
    else:
        new_tag, new_payload = _best_leaf(blk, leaf_ctx)

    # --- 3) 原地替换：重走原树，其余节点字节原样拷贝 ---
    rs, *_ = _read_header(data)
    out = bytearray(data[:4])
    put_varint(out, VERSION)
    put_varint(out, code)
    put_varint(out, h); put_varint(out, w); put_varint(out, min_leaf)

    def rebuild(hh: int, ww: int, gr0: int, gc0: int) -> None:
        p = rs.pos
        tag = rs.data[rs.pos]
        if p == leaf_start and gr0 == lg_r0 and gc0 == lg_c0:   # 命中目标叶子
            rs.pos += 1                                 # 消费旧 tag 字节
            if tag == TAG_UNIFORM:
                rs.varint()
            else:
                ln = rs.varint(); rs.pos += ln
            if new_tag == TAG_UNIFORM:
                out.append(TAG_UNIFORM)
                put_varint(out, zigzag(int(blk[0, 0]) - leaf_ctx))
            else:
                out.append(new_tag)
                put_varint(out, len(new_payload))
                out.extend(new_payload)
            return
        if tag == TAG_QUAD:
            rs.pos += 1
            out.append(TAG_QUAD)
            ch, cw = hh // 2, ww // 2
            for qr0, qr1, qc0, qc1 in ((0, ch, 0, cw), (0, ch, cw, ww),
                                       (ch, hh, 0, cw), (ch, hh, cw, ww)):
                rebuild(qr1 - qr0, qc1 - qc0, gr0 + qr0, gc0 + qc0)
        elif tag == TAG_UNIFORM:
            rs.pos += 1
            out.append(TAG_UNIFORM)
            put_varint(out, rs.varint())
        else:
            rs.pos += 1
            ln = rs.varint()
            out.append(tag)
            put_varint(out, ln)
            out.extend(rs.data[rs.pos:rs.pos + ln])
            rs.pos += ln

    rebuild(h, w, 0, 0)
    payload = bytes(out)
    return payload + zlib.crc32(payload).to_bytes(4, "little")


# ---------------- 自检 ----------------

if __name__ == "__main__":
    rng = np.random.default_rng(7)

    def gen(kind, n):
        if kind == "uniform":
            return np.full((n, n), 7, np.int32)
        if kind == "blocks":
            a = np.zeros((n, n), np.int32)
            for _ in range(40):
                r0, c0 = rng.integers(0, n, 2)
                rh, cw = rng.integers(15, 120, 2)
                a[r0:r0 + rh, c0:c0 + cw] = rng.integers(1, 9)
            return a
        if kind == "smooth":
            i = np.arange(n)
            return (np.sin(np.add.outer(i, i) / 17.0) * 500
                    + np.add.outer(i, i) * 0.7).astype(np.int32)
        if kind == "noise":
            return rng.integers(-1000, 1000, (n, n)).astype(np.int32)
        raise ValueError(kind)

    n = 385                                            # 故意取非 2 的幂、非方块的奇数
    for kind in ("uniform", "blocks", "smooth", "noise"):
        a = gen(kind, n)[:n - 41, :n - 57]             # 非方形尺寸
        enc = encode(a)
        dec = decode(enc)
        assert np.array_equal(dec, a), f"{kind} 无损性失败"
        # 区域解码 == 全图解码的对应切片
        win = decode_region(enc, 100, 260, 37, 199)
        assert np.array_equal(win, a[100:260, 37:199]), f"{kind} 区域解码失败"
        # 局部更新正确性
        b = encode(a)
        b2 = set_pixel(b, 77, 233, -99999)
        a2 = a.copy(); a2[77, 233] = -99999
        assert np.array_equal(decode(b2), a2), f"{kind} 局部更新失败"
        print(f"[OK] {kind:8s}  raw={a.nbytes:>8,}  harc={len(enc):>8,}"
              f"  ratio={a.nbytes / len(enc):6.2f}x")
    print("自检全部通过")
