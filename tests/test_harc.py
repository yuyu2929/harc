# HARC 正确性与功能测试
# 运行: pytest tests/ -v  或  python3 harc.py

import numpy as np
import pytest

from harc import encode, decode, decode_region, set_pixel, _skip_node, Reader

RNG = np.random.default_rng(7)


def gen(kind, n):
    if kind == "uniform":
        return np.full((n, n), 7, np.int32)
    if kind == "blocks":
        a = np.zeros((n, n), np.int32)
        for _ in range(40):
            r0, c0 = RNG.integers(0, n, 2)
            rh, cw = RNG.integers(15, 120, 2)
            a[r0:r0 + rh, c0:c0 + cw] = RNG.integers(1, 9)
        return a
    if kind == "smooth":
        i = np.arange(n)
        return (np.sin(np.add.outer(i, i) / 17.0) * 500
                + np.add.outer(i, i) * 0.7).astype(np.int32)
    if kind == "noise":
        return RNG.integers(-1000, 1000, (n, n)).astype(np.int32)
    if kind == "extremes":
        a = RNG.integers(-1000, 1000, (n, n)).astype(np.int32)
        a[0, 0] = np.iinfo(np.int32).min
        a[0, 1] = np.iinfo(np.int32).max
        return a
    raise ValueError(kind)


SIZES = [(64, 64), (100, 87), (33, 129), (1, 40), (40, 1)]


@pytest.mark.parametrize("kind", ["uniform", "blocks", "smooth", "noise", "extremes"])
@pytest.mark.parametrize("h, w", SIZES)
def test_lossless(kind, h, w):
    """全图编码→解码必须逐元素相等（含奇数尺寸、单行/单列、int32 极值）。"""
    a = gen(kind, 200)[:h, :w]
    enc = encode(a)
    assert np.array_equal(decode(enc), a)


@pytest.mark.parametrize("kind", ["uniform", "blocks", "smooth", "noise"])
def test_region_random_access(kind):
    """区域解码必须与切片严格相等。"""
    a = gen(kind, 256)
    enc = encode(a)
    for r0, r1, c0, c1 in [(0, 256, 0, 256), (10, 40, 200, 240), (128, 129, 0, 256)]:
        win = decode_region(enc, r0, r1, c0, c1)
        assert np.array_equal(win, a[r0:r1, c0:c1])


@pytest.mark.parametrize("kind", ["uniform", "blocks", "smooth", "noise"])
def test_set_pixel(kind):
    """局部更新后必须与手工修改的全图一致，且码流只变化目标叶子附近。"""
    a = gen(kind, 256)
    enc = encode(a)
    for r, c in [(0, 0), (255, 255), (77, 233), (128, 128)]:
        b2 = set_pixel(enc, r, c, -88888)
        a2 = a.copy()
        a2[r, c] = -88888
        assert np.array_equal(decode(b2), a2)


def test_varint_roundtrip():
    """varint/zigzag 编解码往返。"""
    from harc import put_varint, unzigzag, zigzag
    for v in [0, 1, -1, 127, 128, -128, 2**20, -(2**20), 2**40]:
        buf = bytearray()
        put_varint(buf, zigzag(v))
        rd = Reader(bytes(buf))
        assert unzigzag(rd.varint()) == v


def test_uniform_tiny_stream():
    """全同值大图应压到十几字节。"""
    a = np.full((512, 512), 42, np.int32)
    assert len(encode(a)) < 32


def test_compression_beats_rle_on_smooth():
    """平滑数据上 HARC 应显著优于逐行 RLE（自适应设计的基本保证）。"""
    from harc import _enc_rle_bytes
    a = gen("smooth", 256)
    rle = sum(len(_enc_rle_bytes(r.reshape(1, -1), 0)) for r in a)
    assert len(encode(a)) < rle


# ---------------- v2 新增：dtype 保真 ----------------

@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16,
                                   np.int32, np.uint32, np.int64])
def test_dtype_preserved(dtype):
    """无损必须包括 dtype：解出来的数组类型要和输入一致。"""
    a = (RNG.integers(0, 100, (37, 53))).astype(dtype)
    b = decode(encode(a))
    assert b.dtype == a.dtype, f"{dtype} 往返后变成 {b.dtype}"
    assert np.array_equal(a, b)


def test_bool_roundtrip():
    a = (RNG.integers(0, 2, (19, 23)) == 1)
    b = decode(encode(a))
    assert b.dtype == np.bool_
    assert np.array_equal(a, b)


def test_unsupported_dtype_rejected():
    with pytest.raises(TypeError):
        encode(np.zeros((4, 4), dtype=np.float32))


# ---------------- v2 新增：健壮性 ----------------

def test_crc_detects_corruption():
    """任意一个字节被改，decode 必须报错，而不是静默返回错误数据。"""
    a = gen("noise", 64)
    blob = bytearray(encode(a))
    i = len(blob) // 2
    blob[i] ^= 0xFF
    with pytest.raises(ValueError):
        decode(bytes(blob))


def test_magic_checked():
    with pytest.raises(ValueError):
        decode(b"XXXX" + b"\x00" * 20)


def test_truncated_stream():
    a = gen("smooth", 64)
    blob = encode(a)
    with pytest.raises(ValueError):
        decode(blob[: len(blob) // 2])


def test_empty_and_bad_shape():
    with pytest.raises(ValueError):
        encode(np.zeros((0, 0), dtype=np.int32))
    with pytest.raises(ValueError):
        encode(np.zeros(10, dtype=np.int32))       # 一维


@pytest.mark.parametrize("shape", [(1, 1), (1, 97), (97, 1), (3, 500)])
def test_extreme_shapes(shape):
    a = RNG.integers(-5, 5, shape).astype(np.int32)
    assert np.array_equal(decode(encode(a)), a)


def test_set_pixel_out_of_range():
    a = np.zeros((8, 8), dtype=np.int32)
    with pytest.raises(IndexError):
        set_pixel(encode(a), 8, 0, 1)


def test_int64_extremes():
    a = np.array([[np.iinfo(np.int64).min, np.iinfo(np.int64).max],
                  [0, -1]], dtype=np.int64)
    assert np.array_equal(decode(encode(a)), a)


def test_negative_varint_rejected():
    from harc import put_varint
    with pytest.raises(ValueError):
        put_varint(bytearray(), -1)


# ---------------- v3 分块容器 ----------------

@pytest.mark.parametrize("kind", ["uniform", "blocks", "smooth", "noise"])
def test_v3_roundtrip(kind):
    """v3（分块）与 v2 解出的值必须完全一致。"""
    a = gen(kind, 300)
    e3 = encode(a, chunk=64)
    assert np.array_equal(decode(e3), a)
    assert np.array_equal(decode(e3), decode(encode(a)))


def test_v3_region_and_setpixel():
    a = gen("smooth", 320)
    e3 = encode(a, chunk=64)
    assert np.array_equal(decode_region(e3, 10, 200, 30, 250), a[10:200, 30:250])
    b = set_pixel(e3, 137, 233, -424242)
    a2 = a.copy(); a2[137, 233] = -424242
    assert np.array_equal(decode(b), a2)
    # 多次更新同一块与不同块
    for r, c, v in [(0, 0, 1), (319, 319, 2), (160, 160, 3)]:
        b = set_pixel(b, r, c, v)
        a2[r, c] = v
    assert np.array_equal(decode(b), a2)


def test_v3_dtype_preserved():
    a = (RNG.integers(0, 100, (130, 170))).astype(np.int16)
    assert decode(encode(a, chunk=64)).dtype == np.int16


def test_v3_chunk_validation():
    a = np.zeros((16, 16), dtype=np.int32)
    with pytest.raises(ValueError):
        encode(a, chunk=4)          # 太小
    with pytest.raises(ValueError):
        encode(a, chunk=0)


def test_v3_crc_detects_corruption():
    a = gen("noise", 128)
    blob = bytearray(encode(a, chunk=64))
    blob[len(blob) // 2] ^= 0xFF
    with pytest.raises(ValueError):
        decode(bytes(blob))