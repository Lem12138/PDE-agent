# evaluate.py — 测试集自回归推理，生成 task{N}_pred.hdf5 + task{N}_time.csv + 比赛分数
import argparse, os, sys, time, json
import numpy as np
import torch
import h5py

# Workspace on path for model import
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

# Defaults (overridable via data_config.json)
T_IN = 10          # 输入时间窗口
T_PREDICT = 190    # 预测步数
T_TOTAL = 200      # 总时间步
N_SPATIAL = 256    # 目标空间分辨率

def _load_data_config(output_dir):
    """Load data adaptation config if present."""
    cfg_path = os.path.join(output_dir, 'data_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            return json.load(f)
    return {}

def _detect_model_interface(model, device, N):
    """Detect whether model needs grid input (FNO) or plain."""
    x = torch.randn(2, N, 10, device=device)
    try:
        model(x)
        return 'plain'
    except TypeError:
        grid = torch.linspace(0, 1, N, device=device).view(1, N, 1).expand(2, N, 1)
        model(x, grid)
        return 'grid'


# ====================================================================
# 比赛评分公式
# ====================================================================

def compute_rel_mse(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    计算分段 Rel-MSE 和比赛总分。
    pred, gt: (N, T_TOTAL, N_SPATIAL)
    仅评估预测步 [T_IN:]。
    """
    # 逐样本逐步: sum_over_space((p-g)²) / sum_over_space(g²)
    num = ((pred[:, T_IN:, :] - gt[:, T_IN:, :]) ** 2).sum(-1)       # (N, T_pred)
    den = (gt[:, T_IN:, :] ** 2).sum(-1) + 1e-8                       # (N, T_pred)
    rel_mse_per_step = num / den                                       # (N, T_pred)
    rel_mse_per_step = rel_mse_per_step.clip(max=5.0)                  # 单样本上限

    # 对所有样本取均值 → per-step Rel-MSE
    per_step_mean = rel_mse_per_step.mean(axis=0)                      # (T_pred,)

    seg1 = float(per_step_mean[:47].mean())
    seg2 = float(per_step_mean[47:95].mean())
    seg3 = float(per_step_mean[95:].mean())
    overall = float(per_step_mean.mean())

    return {"seg1": seg1, "seg2": seg2, "seg3": seg3, "overall": overall}


def compute_competition_score(metrics: dict, train_time_s: float = 0,
                              infer_time_s: float = 0) -> dict:
    """
    根据比赛公式计算得分（Task 1: 满分 150）。
    - 预测分 (0-75): 分段加权 × 0.75
    - 训练耗时分 (0-35): ≤60min=35, ≤120min=25, ≤300min=20, ≤500min=10, >500min=0
    - 推理耗时分 (0-40): ≤1min=40, 1-2min 线性递减, >2min=0
    """
    seg1, seg2, seg3 = metrics["seg1"], metrics["seg2"], metrics["seg3"]

    # 分段预测得分
    score1 = 100 * np.exp(-20 * seg1)
    score2 = 100 * np.exp(-10 * seg2)
    rmse = np.sqrt(seg3)
    score3 = 100 / (1 + 10 * rmse)   # Lorentzian

    pred_score_base = 0.25 * score1 + 0.25 * score2 + 0.5 * score3
    pred_score = pred_score_base * 0.75  # 满分 75

    # 训练耗时分
    train_min = train_time_s / 60.0
    if train_min <= 60:
        train_score = 35
    elif train_min <= 120:
        train_score = 25
    elif train_min <= 300:
        train_score = 20
    elif train_min <= 500:
        train_score = 10
    else:
        train_score = 0

    # 推理耗时分
    infer_min = infer_time_s / 60.0
    if infer_min <= 1.0:
        infer_score = 40
    elif infer_min <= 2.0:
        infer_score = 40 * (2.0 - infer_min) / 1.0  # 线性递减
    else:
        infer_score = 0

    total = pred_score + train_score + infer_score
    return {
        "pred_score": round(pred_score, 2),
        "train_score": train_score,
        "infer_score": round(infer_score, 2),
        "total": round(total, 2),
        "max_total": 150,
        "seg1_score": round(score1, 2),
        "seg2_score": round(score2, 2),
        "seg3_score": round(score3, 2),
    }


def load_model(checkpoint_path: str):
    """Load and return model from checkpoint. Tries model.py class first, falls back to checkpoint inference."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = raw
    for key in ["model_state_dict", "state_dict", "model"]:
        if key in raw:
            state = raw[key]
            break

    # Try to use the Model class from model.py
    try:
        import model as _m
        classes = [v for v in vars(_m).values()
                   if isinstance(v, type) and issubclass(v, torch.nn.Module)
                   and v.__module__ == 'model']
        if classes:
            ModelClass = classes[0]
        else:
            for name in ['UNet1d', 'FNO1d', 'DeepONet1d', 'MGNO1d', 'Model']:
                obj = getattr(_m, name, None)
                if obj is not None and isinstance(obj, type):
                    ModelClass = obj
                    break
            else:
                raise RuntimeError("No model class found")

        # Try to instantiate with state-dict-inferred params
        width = state[list(state.keys())[0]].shape[0]
        model = ModelClass(modes=12, width=width, in_channels=10, out_channels=1)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    except Exception:
        pass

    # Fallback: FNO-style model
    from model import Model
    width = state[list(state.keys())[0]].shape[0]
    modes = None
    for k, v in state.items():
        if "conv" in k and "weight" in k and hasattr(v, "is_complex") and v.is_complex():
            modes = v.shape[-1]
            break
    in_channels = 10
    n_layers = sum(1 for k in state if k.startswith("conv") and "weights1" in k)
    if n_layers == 0:
        n_layers = 4

    model = Model(modes=modes or 12, width=width, in_channels=in_channels,
                  out_channels=1, n_layers=n_layers)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def autoregressive_rollout(model, test_data: np.ndarray, device: torch.device,
                           t_in: int = 10, pred_steps: int = 190, iface: str = "plain",
                           n_spatial: int = 256) -> np.ndarray:
    """
    对每个测试样本执行自回归 rollout。
    输入: test_data shape (N, T_input, N_SPATIAL) — T_input >= t_in
    输出: pred shape (N, t_in + pred_steps, N_SPATIAL)
    前 t_in 步 = ground truth，后面 pred_steps 步 = 模型预测
    """
    N = test_data.shape[0]
    T_total_out = t_in + pred_steps
    preds_all = np.zeros((N, T_total_out, n_spatial), dtype=np.float32)

    with torch.no_grad():
        for i in range(N):
            sample = torch.from_numpy(test_data[i]).float().to(device)  # (T, X)
            # Copy initial window from GT
            copy_len = min(t_in, sample.shape[0])
            preds_all[i, :copy_len, :] = test_data[i, :copy_len, :]

            window = sample[:t_in].permute(1, 0).unsqueeze(0)  # (1, X, t_in)

            for t in range(t_in, T_total_out):
                if iface == "grid":
                    grid = torch.linspace(0, 1, n_spatial, device=device).view(1, n_spatial, 1)
                    out = model(window, grid)
                else:
                    out = model(window)

                if isinstance(out, tuple):
                    out = out[0]
                out = out.squeeze()
                if out.dim() == 1:
                    out = out  # (X,)
                elif out.dim() == 2:
                    out = out.squeeze(0)  # (1, X) -> (X,)

                preds_all[i, t, :] = out.cpu().numpy()

                next_step = out.unsqueeze(0).unsqueeze(-1)  # (1, X, 1)
                window = torch.cat([window[:, :, 1:], next_step], dim=-1)

            if (i + 1) % 100 == 0:
                print(f"  Inference: {i+1}/{N} samples done")

    return preds_all


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--test_data", required=True, help="Path to test HDF5")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--no_gt", action="store_true", help="Test data has no ground truth (only initial conditions)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load data config (from Captain) ----
    data_cfg = _load_data_config(args.output_dir)
    h5_key = data_cfg.get('h5_key', 'tensor')
    subsample = data_cfg.get('spatial_subsample', 4)
    t_in = data_cfg.get('t_in', T_IN)
    n_spatial = data_cfg.get('n_spatial', N_SPATIAL)
    # Auto-detect no_gt from config (Captain sets this during probe)
    no_gt = no_gt or not data_cfg.get('test_has_gt', True)

    # ---- Load model ----
    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: {type(model).__name__}, params={param_count:,}, device={device}")

    # ---- Load test data ----
    print(f"Loading test data: {args.test_data}")
    with h5py.File(args.test_data, "r") as f:
        for key in [h5_key, "tensor", "data", "u"]:
            if key in f:
                test_raw = f[key][:]
                break
    print(f"  Test shape: {test_raw.shape}")

    # Spatial resampling
    if subsample > 1 and test_raw.shape[-1] > n_spatial:
        test_raw = test_raw[:, :, ::subsample]
    if test_raw.shape[-1] != n_spatial:
        test_tensor = torch.from_numpy(test_raw).float()
        test_tensor = torch.nn.functional.interpolate(
            test_tensor.permute(0, 2, 1), size=n_spatial, mode="linear"
        ).permute(0, 2, 1)
        test_raw = test_tensor.numpy()
    print(f"  After resample: {test_raw.shape}")

    # ---- Detect model interface ----
    iface = _detect_model_interface(model, device, test_raw.shape[-1])
    print(f"  Model interface: {iface}")

    # ---- Inference ----
    pred_steps = test_raw.shape[1] - t_in if not no_gt else T_PREDICT
    if pred_steps < 1 and not no_gt:
        pred_steps = T_PREDICT
    print(f"Running autoregressive rollout ({pred_steps} steps x {test_raw.shape[0]} samples)...")
    t0 = time.time()
    preds = autoregressive_rollout(model, test_raw, device, t_in=t_in, pred_steps=pred_steps,
                                   iface=iface, n_spatial=test_raw.shape[-1])
    infer_time = time.time() - t0
    print(f"  Inference done: {infer_time:.1f}s")

    # ---- Save predictions ----
    pred_path = os.path.join(args.output_dir, f"task{args.task_id}_pred.hdf5")
    with h5py.File(pred_path, "w") as f:
        f.create_dataset("tensor", data=preds, compression="gzip")
    print(f"  Predictions saved: {pred_path} ({os.path.getsize(pred_path)/1024/1024:.1f} MB)")

    # ---- Save timing ----
    csv_path = os.path.join(args.output_dir, f"task{args.task_id}_time.csv")
    with open(csv_path, "w") as f:
        f.write("train_time,inference_time\n")
        f.write(f"0,{infer_time:.2f}\n")
    print(f"  Timing saved: {csv_path}")

    # ---- Scoring (only if GT available) ----
    if no_gt:
        print("  --no_gt set: skipping competition scoring.")
    elif test_raw.shape[1] >= t_in + 10:
        print(f"\n{'='*50}")
        print("  Competition Score Calculation")
        print(f"{'='*50}")
        metrics = compute_rel_mse(preds, test_raw)
        scores = compute_competition_score(metrics, train_time_s=0, infer_time_s=infer_time)
        print(f"  Seg1 Rel-MSE: {metrics['seg1']:.6f}  -> Score: {scores['seg1_score']:.2f}")
        print(f"  Seg2 Rel-MSE: {metrics['seg2']:.6f}  -> Score: {scores['seg2_score']:.2f}")
        print(f"  Seg3 Rel-MSE: {metrics['seg3']:.6f}  -> Score: {scores['seg3_score']:.2f}")
        print(f"  Overall:      {metrics['overall']:.6f}")
        print(f"  ---")
        print(f"  Pred  Score: {scores['pred_score']:.2f}/75")
        print(f"  Train Score: {scores['train_score']}/35")
        print(f"  Infer Score: {scores['infer_score']:.2f}/40")
        print(f"  TOTAL:       {scores['total']:.2f}/150")
        score_path = os.path.join(args.output_dir, f"task{args.task_id}_score.json")
        with open(score_path, "w") as f:
            json.dump({"metrics": metrics, "scores": scores, "infer_time_s": infer_time_s}, f, indent=2)
    else:
        print(f"  Test data has only {test_raw.shape[1]} time steps; skipping scoring.")

    print("Done.")


if __name__ == "__main__":
    main()
