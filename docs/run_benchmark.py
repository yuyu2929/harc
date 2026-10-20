# -*- coding: utf-8 -*-
"""HARC 基准测试（可复现 docs/benchmark.md 的全部数据）
运行: python3 docs/run_benchmark.py
输出: 终端表格 + docs/benchmark_chart.png
"""
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harc import encode, decode, decode_region, set_pixel, _enc_rle_bytes  # noqa: E402

N = 512


def make_datasets():
    rng = np.random.default_rng(42)
    ds = {}
    ds["均质地物"] = np.full((N, N), 7, np.int32)

    a = np.zeros((N, N), np.int32)
    for _ in range(50):
        r0, c0 = rng.integers(0, N, 2)
        rh, cw = rng.integers(20, 160, 2)
        a[r0:r0 + rh, c0:c0 + cw] = rng.integers(1, 9)
    ds["大斑块地类图"] = a

    i = np.arange(N)
    ds["平滑DEM"] = (np.sin(np.add.outer(i, i) / 17.0) * 500
                     + np.add.outer(i, i) * 0.7).astype(np.int32)
    ds["随机噪声"] = rng.integers(-1000, 1000, (N, N)).astype(np.int32)

    m = np.zeros((N, N), np.int32)
    m[:N // 2, :N // 2] = ds["大斑块地类图"][:N // 2, :N // 2]
    m[:N // 2, N // 2:] = ds["平滑DEM"][:N // 2, N // 2:]
    m[N // 2:, :N // 2] = rng.integers(-100, 100, (N // 2, N // 2))
    m[N // 2:, N // 2:] = 5
    ds["混合场景"] = m
    return ds


def rle_size(a):
    return sum(len(_enc_rle_bytes(r.reshape(1, -1), 0)) for r in a)


def tiled_zlib_size(a, tile=64):
    return sum(len(zlib.compress(a[r:r + tile, c:c + tile].tobytes(), 6))
               for r in range(0, a.shape[0], tile)
               for c in range(0, a.shape[1], tile))


# ---------------- 与业界格式对比（可选依赖，缺失则跳过） ----------------

def lerc_size(a):
    """Esri LERC 无损（maxZErr=0）。返回字节数，不可用返回 None。"""
    try:
        import lerc
    except Exception:
        return None
    try:
        arr = np.ascontiguousarray(a)
        r = lerc.encode(arr, 1, False, None, 0, arr.nbytes)   # (status, nBytes, buf)
        return int(r[1])
    except Exception:
        return None


def zarr_size(a, tile=64):
    """Zarr v3（Zstd 压缩，64×64 分块）。返回落盘总字节（含元数据），不可用返回 None。"""
    try:
        import os
        import shutil
        import tempfile

        import zarr
        import zarr.codecs as zc
    except Exception:
        return None
    tmp = None
    try:
        tmp = tempfile.mkdtemp(prefix="harc-zarr-")
        kw = dict(store=zarr.storage.LocalStore(tmp), shape=a.shape,
                  dtype=a.dtype, chunks=(tile, tile), overwrite=True)
        if hasattr(zc, "ZstdCodec"):
            kw["compressors"] = [zc.ZstdCodec()]
        arr = zarr.create_array(**kw)
        arr[:] = a
        total = 0
        for root, _dirs, files in os.walk(tmp):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total
    except Exception:
        return None
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def compare_table(datasets):
    """HARC vs LERC vs Zarr，三者都跑不到就跳过整张表。"""
    rows = []
    for name, a in datasets.items():
        rows.append((name, a.nbytes, lerc_size(a), zarr_size(a), len(encode(a))))
    if all(r[2] is None for r in rows) and all(r[3] is None for r in rows):
        print("\n（LERC / Zarr 均不可用，跳过对比表）")
        return
    print(f"\n{'数据集':<12}{'原始':>10}{'LERC':>9}{'比率':>8}{'Zarr':>9}{'比率':>8}{'HARC':>9}{'比率':>8}")
    print("-" * 78)
    for name, raw, l, z, h in rows:
        ls = f"{l:,}" if l else "-"
        zs = f"{z:,}" if z else "-"
        lr = f"{raw / l:.1f}x" if l else "-"
        zr = f"{raw / z:.1f}x" if z else "-"
        print(f"{name:<12}{raw:>10,}{ls:>9}{lr:>8}{zs:>9}{zr:>8}{h:>9,}{raw / h:>7.1f}x")


def main():
    datasets = make_datasets()
    print(f"{'数据集':<12}{'原始':>10}{'RLE':>9}{'zlib':>9}{'块zlib':>9}"
          f"{'HARC':>9}{'比率':>10}{'编码ms':>8}{'解码ms':>8}")
    print("-" * 90)
    for name, a in datasets.items():
        raw = a.nbytes
        t = time.perf_counter(); enc = encode(a); t_enc = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); dec = decode(enc); t_dec = (time.perf_counter() - t) * 1000
        assert np.array_equal(dec, a), f"{name} 无损性失败"
        win = decode_region(enc, 200, 300, 200, 300)
        assert np.array_equal(win, a[200:300, 200:300])
        b2 = set_pixel(enc, 137, 421, -88888)
        a2 = a.copy(); a2[137, 421] = -88888
        assert np.array_equal(decode(b2), a2)
        print(f"{name:<12}{raw:>10,}{rle_size(a):>9,}"
              f"{len(zlib.compress(a.tobytes(), 6)):>9,}{tiled_zlib_size(a):>9,}"
              f"{len(enc):>9,}{raw / len(enc):>9.1f}x{t_enc:>8.0f}{t_dec:>8.0f}")

    compare_table(datasets)


if __name__ == "__main__":
    main()
