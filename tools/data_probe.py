# data_probe.py — 数据探测工具，自动分析HDF5数据和checkpoint，生成物理特征报告供Captain推理
import os
import numpy as np
import h5py
import torch


def probe_hdf5(filepath: str, max_samples: int = 5) -> dict:
    """
    探测 HDF5 数据文件，提取元信息、统计量和频谱特征。
    返回的 dict 可直接写入 ResourcePool。
    """
    if not os.path.exists(filepath):
        return {"error": f"文件不存在: {filepath}"}

    with h5py.File(filepath, "r") as f:
        # 找到主数据集
        # 常见键名：'tensor', 'data', 'u', 'sol'
        data_key = None
        for candidate in ["tensor", "data", "u", "sol"]:
            if candidate in f:
                data_key = candidate
                break
        if data_key is None:
            # 取第一个非 group 的 dataset
            for k in f.keys():
                if isinstance(f[k], h5py.Dataset):
                    data_key = k
                    break
        if data_key is None:
            return {"error": f"未找到数据集，可用键: {list(f.keys())}"}

        dset = f[data_key]
        shape = dset.shape
        dtype = str(dset.dtype)

        # 采样分析（避免加载全量 7GB 数据）
        n_samples = min(max_samples, shape[0])
        indices = np.linspace(0, shape[0] - 1, n_samples, dtype=int)
        samples = dset[indices].astype(np.float64)

    # ---- 统计量 ----
    v_min = float(samples.min())
    v_max = float(samples.max())
    v_mean = float(samples.mean())
    v_std = float(samples.std())
    has_nan = bool(np.any(np.isnan(samples)))
    has_inf = bool(np.any(np.isinf(samples)))

    # ---- 推断数据维度 ----
    # shape: (N, T, X) 或 (N, X, T) — 找哪个轴是时间（步数多）、哪个是空间（值平滑变化）
    dim_analysis = _infer_dimensions(shape, samples)

    # ---- 频谱分析 ----
    freq_analysis = _frequency_analysis(samples, dim_analysis)

    # ---- 边界条件检测 ----
    bc_analysis = _check_boundary(samples, dim_analysis)

    # ---- 时间演化 ----
    temporal_analysis = _temporal_evolution(samples, dim_analysis)

    return {
        "file": os.path.basename(filepath),
        "shape": list(shape),
        "dtype": dtype,
        "statistics": {
            "min": v_min, "max": v_max, "mean": v_mean, "std": v_std,
            "has_nan": has_nan, "has_inf": has_inf,
        },
        "dimensions": dim_analysis,
        "frequency": freq_analysis,
        "boundary": bc_analysis,
        "temporal": temporal_analysis,
    }


def probe_checkpoint(filepath: str) -> dict:
    """
    探测 PyTorch checkpoint，反推模型架构参数。
    返回的 dict 可直接写入 ResourcePool。
    """
    if not os.path.exists(filepath):
        return {"error": f"文件不存在: {filepath}"}

    try:
        raw = torch.load(filepath, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"error": f"加载失败: {e}"}

    # 解包外层键
    outer_key = None
    state_dict = raw
    if isinstance(raw, dict):
        for candidate in ["model_state_dict", "state_dict", "model"]:
            if candidate in raw:
                state_dict = raw[candidate]
                outer_key = candidate
                break

    if not isinstance(state_dict, dict) or len(state_dict) < 3:
        return {"error": f"无法解析 state_dict，键数: {len(state_dict) if isinstance(state_dict, dict) else 'N/A'}"}

    # 从权重形状反推参数
    params = {}
    try:
        # fc0: lifting layer (in_channels → width)
        if "fc0.weight" in state_dict:
            fc0_w = state_dict["fc0.weight"]
            params["width"] = fc0_w.shape[0]
            params["fc0_input_dim"] = fc0_w.shape[1]  # 含 grid 坐标
            params["in_channels"] = fc0_w.shape[1] - 1  # 减掉 grid 维

        # fc2: output projection → out_channels
        if "fc2.weight" in state_dict:
            params["out_channels"] = state_dict["fc2.weight"].shape[0]

        # fc1: intermediate projection
        if "fc1.weight" in state_dict:
            params["fc1_output"] = state_dict["fc1.weight"].shape[0]

        # SpectralConv: modes from complex weights
        import builtins
        for name, tensor in state_dict.items():
            if tensor.dtype in (getattr(torch, "complex64", None), getattr(torch, "complex128", None)):
                pass
        modes = None
        for k, v in state_dict.items():
            if "conv" in k and "weight" in k:
                try:
                    if v.is_complex():
                        modes = v.shape[-1]
                        break
                except Exception:
                    pass
        params["modes"] = modes

        # n_layers: count conv{i}.weights1 keys
        n = 0
        while f"conv{n}.weights1" in state_dict or f"convs.{n}.weights1" in state_dict:
            n += 1
        if n == 0:
            # Try encoder-based naming (UNet)
            e = 1
            while f"encoder{e}.enc{e}conv1.weight" in state_dict:
                e += 1
            params["n_encoders"] = e - 1 if e > 1 else None
        params["n_layers"] = n if n > 0 else None

        # 完整键列表（前20个）
        params["state_keys_sample"] = sorted(state_dict.keys())[:20]
    except Exception as e:
        params["parse_error"] = str(e)

    return {
        "file": os.path.basename(filepath),
        "size_kb": round(os.path.getsize(filepath) / 1024, 1),
        "outer_key": outer_key,
        "total_keys": len(state_dict),
        "params": params,
    }


# ====================================================================
# 内部分析函数
# ====================================================================

def _infer_dimensions(shape: tuple, samples: np.ndarray) -> dict:
    """推断数据各轴含义：样本/时间/空间。"""
    ndim = len(shape)
    if ndim == 3:
        # 启发式 1: 点数多的轴通常是空间（X >> T）
        # 启发式 2: 空间相邻点差分渐进，时间相邻步可能因激波跳跃
        size_mid, size_last = shape[1], shape[2]
        g_mid = float(np.abs(np.diff(samples, axis=1)).mean())
        g_last = float(np.abs(np.diff(samples, axis=2)).mean())

        # 空间信号：点数多 + 相邻差分相对平稳 → 梯度密度低
        grad_density_mid = g_mid / max(size_mid, 1)
        grad_density_last = g_last / max(size_last, 1)

        # 点数优先，梯度密度辅助
        if size_last >= size_mid * 4 or (size_last > size_mid and grad_density_last < grad_density_mid):
            return {"structure": "N_samples × T_steps × X_space",
                    "n_samples": shape[0], "n_timesteps": shape[1], "n_spatial": shape[2]}
        else:
            return {"structure": "N_samples × X_space × T_steps",
                    "n_samples": shape[0], "n_spatial": shape[1], "n_timesteps": shape[2]}
    elif ndim == 2:
        return {"structure": "N_samples × X_space (2D)", "n_samples": shape[0], "n_spatial": shape[1]}
    else:
        return {"structure": f"Unknown {ndim}D", "shape": list(shape)}


def _frequency_analysis(samples: np.ndarray, dims: dict) -> dict:
    """对空间维做 FFT，分析频率成分分布。"""
    n_spatial = dims.get("n_spatial", samples.shape[-1])
    spatial_axis = -1  # 假设最后一维是空间

    # 沿空间维做 FFT
    fft = np.abs(np.fft.rfft(samples, axis=spatial_axis))
    # 取第一个样本、中间时间步
    if samples.ndim == 3:
        mid_t = samples.shape[-2] // 2 if "T_steps" in dims.get("structure", "") else 0
        fft_slice = fft[0, mid_t, :] if samples.ndim == 3 else fft[0, :]
    else:
        fft_slice = fft[0, :]

    total_energy = float(np.sum(fft_slice ** 2))
    if total_energy > 1e-30:
        cumsum = np.cumsum(fft_slice ** 2) / total_energy
        # 50% 和 90% 能量带宽
        bw50 = int(np.searchsorted(cumsum, 0.5))
        bw90 = int(np.searchsorted(cumsum, 0.9))
    else:
        bw50 = bw90 = 0

    # 高频占比：后 1/3 频率模态的能量占比
    n_modes = len(fft_slice)
    high_energy_ratio = float(np.sum(fft_slice[2 * n_modes // 3:] ** 2) / (total_energy + 1e-30))

    interpretation = "smooth_dominated" if high_energy_ratio < 0.05 else (
        "mixed" if high_energy_ratio < 0.15 else "shock_rich")

    return {
        "high_freq_energy_ratio": round(high_energy_ratio, 4),
        "energy_bw50_modes": bw50,
        "energy_bw90_modes": bw90,
        "interpretation": interpretation,
        "note": {
            "smooth_dominated": "低频主导，解场平滑，FNO 少量 modes 即可覆盖",
            "mixed": "中等高频成分，需要适中的 modes/分辨率",
            "shock_rich": "高频成分显著，存在激波/间断，需要强局部建模能力（UNet 优于 FNO）",
        }.get(interpretation, ""),
    }


def _check_boundary(samples: np.ndarray, dims: dict) -> dict:
    """检测空间边界条件类型。"""
    spatial_axis = -1
    # 首尾空间点差值
    if samples.ndim == 3:
        edge_0 = samples[:, :, 0]
        edge_N = samples[:, :, -1]
    else:
        edge_0 = samples[:, 0]
        edge_N = samples[:, -1]

    diff = float(np.abs(edge_0 - edge_N).mean())
    amplitude = float(np.abs(samples).std())
    relative_gap = diff / (amplitude + 1e-8)

    # 周期边界：首尾值接近 → relative_gap < 0.1
    # Dirichlet：首尾都接近 0
    zero_tolerance = amplitude * 0.05
    is_periodic = relative_gap < 0.1
    is_dirichlet = (float(np.abs(edge_0).mean()) < zero_tolerance and
                    float(np.abs(edge_N).mean()) < zero_tolerance)

    bc_type = "periodic" if is_periodic else ("dirichlet" if is_dirichlet else "other")
    return {
        "type": bc_type,
        "edge_diff_relative": round(relative_gap, 4),
        "note": {
            "periodic": "周期边界 → FNO 天然契合（FFT 无 padding 误差）",
            "dirichlet": "Dirichlet 边界 → FNO 需要 padding，UNet 不受限",
            "other": "非标准边界 → 需具体分析",
        }.get(bc_type, ""),
    }


def _temporal_evolution(samples: np.ndarray, dims: dict) -> dict:
    """分析时间演化：解是收敛、发散还是震荡。"""
    if samples.ndim < 3 or "n_timesteps" not in dims:
        return {"note": "单时间步或无时间维，跳过"}

    # 取第一个样本，沿时间跟踪空间均值
    trace = samples[0, :, :].mean(axis=-1)  # (T,)
    t_start = float(trace[:5].mean())
    t_end = float(trace[-5:].mean())
    trend = "growing" if t_end > t_start * 1.1 else (
        "decaying" if t_end < t_start * 0.9 else "stable")

    # 时间序列的震荡程度
    oscillation = float(np.std(np.diff(trace)))

    return {
        "trend": trend,
        "spatial_mean_start": round(t_start, 6),
        "spatial_mean_end": round(t_end, 6),
        "oscillation": round(oscillation, 6),
    }


# ====================================================================
# 报告格式化
# ====================================================================

def format_probe_report(data_probes: list, ckpt_probes: list) -> str:
    """将所有探测结果格式化为 Markdown，直接注入 Captain prompt。"""
    lines = ["## 自动探测报告\n"]

    # 数据部分
    lines.append("### 数据探测\n")
    for r in data_probes:
        if "error" in r:
            lines.append(f"- ❌ {r['file']}: {r['error']}\n")
            continue
        lines.append(f"**{r['file']}**")
        dims = r.get("dimensions", {})
        lines.append(f"- 维度: {r['shape']} ({dims.get('structure', '?')}), dtype={r['dtype']}")
        stats = r.get("statistics", {})
        lines.append(f"- 统计: min={stats['min']:.4f}, max={stats['max']:.4f}, "
                     f"mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        freq = r.get("frequency", {})
        lines.append(f"- 频谱: 高频能量占比={freq.get('high_freq_energy_ratio', '?')}, "
                     f"判定={freq.get('interpretation', '?')} → {freq.get('note', '')}")
        bc = r.get("boundary", {})
        lines.append(f"- 边界: {bc.get('type', '?')} → {bc.get('note', '')}")
        temp = r.get("temporal", {})
        if "trend" in temp:
            lines.append(f"- 时间演化: {temp['trend']} (起始={temp['spatial_mean_start']:.4f}, "
                         f"结束={temp['spatial_mean_end']:.4f})")
        lines.append("")

    # Checkpoint 部分
    lines.append("### Checkpoint 探测\n")
    for c in ckpt_probes:
        if "error" in c:
            lines.append(f"- ❌ {c['file']}: {c['error']}\n")
            continue
        p = c.get("params", {})
        lines.append(f"**{c['file']}** ({c['size_kb']}KB, {c['total_keys']} keys)")
        lines.append(f"- 推断参数: width={p.get('width')}, modes={p.get('modes')}, "
                     f"in_channels={p.get('in_channels')}, out_channels={p.get('out_channels')}, "
                     f"n_layers={p.get('n_layers')}")
        if p.get("n_encoders"):
            lines.append(f"- UNet: n_encoders={p['n_encoders']}")
        if p.get("state_keys_sample"):
            lines.append(f"- 键名样本: {p['state_keys_sample'][:8]}")
        lines.append("")

    return "\n".join(lines)


# ====================================================================
# 快速探测入口
# ====================================================================

def scan_checkpoints(ckpt_dir: str, pattern: str = "*.pt") -> list:
    """扫描目录下所有 checkpoint 文件，返回全量探测结果列表。"""
    import glob
    results = []
    for fpath in sorted(glob.glob(os.path.join(ckpt_dir, pattern))):
        results.append(probe_checkpoint(fpath))
    return results
