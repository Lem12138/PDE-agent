#!/usr/bin/env python3
# prepare_data.py — 将 PDEBench 原始 h5 转换为统一的 train/val/test 切分文件。
#
# 支持两种源格式：
#   1. 分组式 group  ：顶层键为 0000/0001/...，每个组含 data [T, X, C]（如 1D diffusion-sorption）
#   2. 键式    tensor：顶层直接有 tensor [N, T, X] 或 [N, T, X, C]（如 2D DarcyFlow）
#
# 提供两种用法：
#   - 作为库：from tools.prepare_data import prepare_split, needs_split
#   - 作为脚本：python tools/prepare_data.py --src <file> --out_dir <dir>
#
# 设计原则：
#   - 源文件以只读方式打开，绝不修改。
#   - 切分是一次 permutation 后互斥分配，保证 train/val/test 无重叠。
#   - 不做归一化、不改变数值精度；仅把 [T, X, 1] 的哑通道维去掉。
#   - 缓存：源 sha256 + 种子 + 比例三项一致时直接复用已有切分。

import os
import json
import time
import hashlib
import argparse

import numpy as np
import h5py


# ---------------------------------------------------------------- 工具函数

def _sha256(path, buf_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def detect_format(f):
    """返回 ('group', [分组名...]) 或 ('tensor', 'tensor')。"""
    keys = list(f.keys())
    if not keys:
        raise RuntimeError("源文件为空")

    if "tensor" in f and isinstance(f["tensor"], h5py.Dataset):
        return "tensor", "tensor"

    k0 = keys[0]
    if isinstance(f[k0], h5py.Group) and "data" in f[k0]:
        groups = [k for k in keys if isinstance(f[k], h5py.Group) and "data" in f[k]]
        groups.sort()
        return "group", groups

    raise RuntimeError(f"无法识别的源格式，顶层键示例: {keys[:5]}")


def read_sample(f, fmt, key, i):
    """读取单个样本，返回去哑通道后的数组。"""
    if fmt == "group":
        arr = f[key[i]]["data"][:]
    else:
        arr = f[key][i]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def needs_split(path):
    """判断一个 h5 文件是否是「未划分」的原始数据。

    规则（纯确定性）：
      - 文件名含 train/val/test → 视为已划分，不需要切
      - 顶层是 tensor 且样本数>1 → 未划分
      - 顶层是 0000 式分组 → 未划分
    返回 (bool, 原因字符串)
    """
    name = os.path.basename(path).lower()
    if any(k in name for k in ("train", "val", "valid", "test")):
        return False, "文件名含 train/val/test，视为已划分"
    try:
        with h5py.File(path, "r") as f:
            fmt, key = detect_format(f)
            n = len(key) if fmt == "group" else f[key].shape[0]
            return True, f"未划分（格式={fmt}, 样本数={n}）"
    except Exception as e:
        return False, f"无法读取: {e}"


# ---------------------------------------------------------------- 核心切分

def prepare_split(src, out_dir, ratios=(0.8, 0.1, 0.1), seed=42,
                  chunk=256, verify=True, reuse=True, log=print):
    """把源文件切成 train/val/test，返回 (manifest, reused)。

    reuse=True 时，若 out_dir 里已有 manifest 且 源sha256/种子/比例 全部一致，
    直接复用已有文件，不重复切分。
    """
    src = os.path.abspath(src)
    if not os.path.exists(src):
        raise FileNotFoundError(f"源文件不存在: {src}")

    tr, va, te = ratios
    if abs(tr + va + te - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须为 1，当前 {tr + va + te}")

    t0 = time.time()
    src_size = os.path.getsize(src)
    log(f"源文件: {src} ({src_size / 1e9:.2f} GB)")
    src_sha = _sha256(src)
    log(f"源 sha256: {src_sha[:16]}...")

    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "split_manifest.json")

    # ---- 缓存检查 ----
    if reuse and os.path.exists(mpath):
        try:
            with open(mpath, "r", encoding="utf-8") as fm:
                old = json.load(fm)
            same = (old.get("source_sha256") == src_sha
                    and old.get("seed") == seed
                    and [round(x, 6) for x in old.get("ratios", [])] == [round(x, 6) for x in ratios]
                    and all(os.path.exists(os.path.join(out_dir, old["files"][k]))
                            for k in ("train", "val", "test")))
            if same:
                log(f"缓存命中，复用已有切分: {out_dir}")
                return old, True
            log("缓存失效（源/种子/比例有变化），重新切分")
        except Exception as e:
            log(f"旧 manifest 无法读取（{e}），重新切分")

    # ---- 探测格式与切分 ----
    with h5py.File(src, "r") as f:          # 只读
        fmt, key = detect_format(f)
        n_total = len(key) if fmt == "group" else f[key].shape[0]
        log(f"格式: {fmt}，样本数: {n_total}")

        sample_shape = read_sample(f, fmt, key, 0).shape
        log(f"单样本形状（去哑通道后）: {sample_shape}")

        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_total)
        n_tr = int(n_total * tr)
        n_va = int(n_total * va)
        idx_tr = perm[:n_tr]
        idx_va = perm[n_tr:n_tr + n_va]
        idx_te = perm[n_tr + n_va:]

        assert len(set(idx_tr) | set(idx_va) | set(idx_te)) == n_total, "索引未覆盖全集"
        assert not (set(idx_tr) & set(idx_va)), "train/val 重叠"
        assert not (set(idx_tr) & set(idx_te)), "train/test 重叠"
        assert not (set(idx_va) & set(idx_te)), "val/test 重叠"
        log(f"切分: train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")

        splits = {"train": idx_tr, "val": idx_va, "test": idx_te}
        out_paths = {}

        for name, idx in splits.items():
            out_path = os.path.join(out_dir, f"{name}.h5")
            out_paths[name] = out_path
            total = len(idx)
            log(f"写入 {name}.h5（{total} 条）...")
            with h5py.File(out_path, "w") as g:
                ds = g.create_dataset(
                    "tensor",
                    shape=(total, *sample_shape),
                    dtype="float32",
                    chunks=(min(chunk, total), *sample_shape),
                )
                for start in range(0, total, chunk):
                    block = idx[start:start + chunk]
                    arr = np.empty((len(block), *sample_shape), dtype=np.float32)
                    for j, src_i in enumerate(block):
                        arr[j] = read_sample(f, fmt, key, int(src_i))
                    ds[start:start + len(block)] = arr
            log(f"  完成 {name}.h5  {os.path.getsize(out_path) / 1e9:.2f} GB")

    # ---- manifest ----
    manifest = {
        "source_file": src,
        "source_size_bytes": src_size,
        "source_sha256": src_sha,
        "source_format": fmt,
        "n_total": n_total,
        "sample_shape": list(sample_shape),
        "dtype": "float32",
        "ratios": [tr, va, te],
        "seed": seed,
        "counts": {k: int(len(v)) for k, v in splits.items()},
        "files": {k: os.path.basename(v) for k, v in out_paths.items()},
        "channel_squeezed": True,
        "notes": "一次 permutation 互斥切分；未做归一化；源文件只读。",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(mpath, "w", encoding="utf-8") as fm:
        json.dump(manifest, fm, ensure_ascii=False, indent=2)
    log(f"manifest: {mpath}")

    # ---- 无损校验 ----
    if verify:
        log("开始无损校验...")
        rng2 = np.random.RandomState(12345)
        ok = True
        with h5py.File(src, "r") as fs:
            for name, idx in splits.items():
                picks = rng2.choice(len(idx), size=min(20, len(idx)), replace=False)
                with h5py.File(out_paths[name], "r") as g:
                    for p in picks:
                        if not np.array_equal(g["tensor"][int(p)],
                                              read_sample(fs, fmt, key, int(idx[p]))):
                            log(f"  ✗ {name}[{p}] 与源不一致")
                            ok = False
                            break
                if ok:
                    log(f"  ✓ {name}: 抽查 20 条与源逐元素一致")
        if not ok:
            raise RuntimeError("无损校验失败")
        log("无损校验通过")

    log(f"全部完成，用时 {time.time() - t0:.1f}s")
    return manifest, False


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="PDEBench 数据切分")
    ap.add_argument("--src", required=True, help="源 h5 文件路径")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"), help="切分比例，默认 0.8 0.1 0.1")
    ap.add_argument("--seed", type=int, default=42, help="打乱种子，默认 42")
    ap.add_argument("--chunk", type=int, default=256, help="写入分块大小")
    ap.add_argument("--no-cache", dest="reuse", action="store_false", default=True,
                    help="忽略缓存，强制重新切分")
    ap.add_argument("--no-verify", dest="verify", action="store_false", default=True,
                    help="跳过无损校验")
    args = ap.parse_args()

    def _log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    manifest, reused = prepare_split(
        src=args.src, out_dir=args.out_dir, ratios=tuple(args.ratios),
        seed=args.seed, chunk=args.chunk, verify=args.verify,
        reuse=args.reuse, log=_log,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n{'复用缓存' if reused else '新建切分'}")


if __name__ == "__main__":
    main()
