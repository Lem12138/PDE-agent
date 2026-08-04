import argparse, os, json, sys, random
import numpy as np, h5py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# ---- Model auto-discovery ----
def _discover_model_class():
    import model as _m
    classes = [v for v in vars(_m).values()
               if isinstance(v, type) and issubclass(v, nn.Module)
               and v.__module__ == 'model']
    ctor_patterns = [
        lambda c: c(),
        lambda c: c(modes=12, width=32, in_channels=10, out_channels=1),
        lambda c: c(modes=12, width=32, in_channels=10, out_channels=1, n_layers=4),
        lambda c: c(modes=12, width=32, in_channels=10, out_channels=1, n_layers=4, n_scales=3),
        lambda c: c(in_channels=10, out_channels=1, base_width=32, depth=4),
        lambda c: c(in_channels=10, out_channels=1, init_features=32),
        lambda c: c(num_channels=1, modes=12, width=32, initial_step=10),
        lambda c: c(in_channels=10, width=32, n_layers=4, n_basis=128, nx=256),
    ]
    for cls in classes:
        for pat in ctor_patterns:
            try:
                pat(cls)
                return cls
            except (TypeError, ValueError):
                continue
    for name in ['UNet1d', 'FNO1d', 'DeepONet1d', 'MGNO1d', 'Model']:
        obj = getattr(_m, name, None)
        if obj is not None and isinstance(obj, type):
            return obj
    if classes:
        return classes[0]
    raise RuntimeError("No nn.Module class found in model.py")

def _detect_model_interface(model, device, N=256, T_IN=10):
    x = torch.randn(2, N, T_IN, device=device)
    try:
        model(x)
        return 'plain'
    except TypeError:
        grid = torch.linspace(0, 1, N, device=device).view(1, N, 1).expand(2, N, 1)
        model(x, grid)
        return 'grid'

def _load_data_config(save_dir):
    cfg_path = os.path.join(save_dir, 'data_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            return json.load(f)
    return {}

# ---------- Dataset ----------
class PDEDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# ---------- Validation ----------
@torch.no_grad()
def validate(model, dataloader, device, T_IN=10, forward_fn=None):
    model.eval()
    all_rel, all_s1, all_s2, all_s3, count = 0.0, 0.0, 0.0, 0.0, 0
    for batch in dataloader:
        batch = batch.to(device)
        B, T, N = batch.shape
        pred_steps = T - T_IN
        if pred_steps < 1:
            continue
        x = batch[:, :T_IN, :].permute(0, 2, 1)
        targets = batch[:, T_IN:, :]
        preds = []
        for _ in range(pred_steps):
            if forward_fn is not None:
                out = forward_fn(x)
            else:
                out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 2:
                out = out.unsqueeze(-1)
            preds.append(out.squeeze(-1))
            x = torch.cat([x[:, :, 1:], out], dim=-1)
        preds = torch.stack(preds, dim=1)
        num = ((preds - targets) ** 2).sum(-1)
        den = (targets ** 2).sum(-1) + 1e-8
        rel = (num / den).clamp(max=5.0)
        s1e = min(47, pred_steps)
        s2e = min(95, pred_steps)
        all_s1 += rel[:, :s1e].mean().item() * B
        all_s2 += rel[:, 47:s2e].mean().item() * B if s2e > 47 else 0
        all_s3 += rel[:, 95:].mean().item() * B if pred_steps > 95 else 0
        all_rel += rel.mean().item() * B
        count += B
    return (all_rel / count, all_s1 / count, all_s2 / count, all_s3 / count)

# ---------- Phase 1: Teacher Forcing ----------
# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--val_data', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--modes', type=int, default=12)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--phase1_lr', type=float, default=1e-4)
    parser.add_argument('--phase1_epochs', type=int, default=50)
    parser.add_argument('--phase2_lr', type=float, default=1e-5)
    parser.add_argument('--phase2_epochs', type=int, default=100)
    parser.add_argument('--max_rollout', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--no_plateau', action='store_true',
                        help='Disable ReduceLROnPlateau + early stopping (for scout mode)')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    T_IN = 10

    # ---- Load data config (written by Captain from probe) ----
    data_cfg = _load_data_config(args.save_dir)
    h5_key = data_cfg.get('h5_key', 'tensor')
    subsample = data_cfg.get('spatial_subsample', 4)

    # ---- Data loading ----
    def _load_h5(path):
        with h5py.File(path, 'r') as f:
            data = f[h5_key][:].astype(np.float32)
        if subsample > 1:
            data = data[:, :, ::subsample]
        return data

    extra_files = data_cfg.get('extra_train_files', [])
    train_raw = _load_h5(args.train_data)
    for ep in extra_files:
        train_raw = np.concatenate([train_raw, _load_h5(ep)], axis=0)
    val_raw = _load_h5(args.val_data)

    if args.max_samples > 0:
        train_raw = train_raw[:args.max_samples]

    mean = train_raw.mean()
    std = train_raw.std() + 1e-6
    train_data = (train_raw - mean) / std
    val_data = (val_raw - mean) / std

    train_loader = DataLoader(PDEDataset(train_data), batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(PDEDataset(val_data), batch_size=16, shuffle=False)

    # ---- Model ----
    N_spatial = train_data.shape[-1]
    ModelClass = _discover_model_class()
    if args.checkpoint:
        raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state = raw.get('model_state_dict', raw.get('state_dict', raw))
        model = ModelClass(modes=args.modes, width=args.width, in_channels=T_IN, out_channels=1)
        model.load_state_dict(state, strict=False)
        model = model.to(device)
    else:
        # Try multiple constructor patterns (same as _discover_model_class)
        ctor_patterns = [
            lambda: ModelClass(modes=args.modes, width=args.width, in_channels=T_IN, out_channels=1),
            lambda: ModelClass(modes=args.modes, width=args.width, in_channels=T_IN, out_channels=1, n_layers=args.n_layers),
            lambda: ModelClass(modes=args.modes, width=args.width, in_channels=T_IN, out_channels=1, n_layers=args.n_layers, n_scales=3),
            lambda: ModelClass(num_channels=1, modes=args.modes, width=args.width, initial_step=T_IN),
            lambda: ModelClass(in_channels=T_IN, out_channels=1, base_width=args.width, depth=args.n_layers),
            lambda: ModelClass(in_channels=T_IN, out_channels=1, init_features=args.width),
            lambda: ModelClass(in_channels=T_IN, width=args.width, n_layers=args.n_layers, n_basis=128, nx=N_spatial),
            lambda: ModelClass(),
        ]
        model = None
        for pat in ctor_patterns:
            try:
                model = pat()
                break
            except (TypeError, ValueError):
                continue
        if model is None:
            raise RuntimeError(f"Cannot instantiate {ModelClass.__name__} with any known pattern")
        model = model.to(device)
    print(f"Model: {ModelClass.__name__}, params={sum(p.numel() for p in model.parameters()):,}")

    # Detect model interface (FNO needs grid, others don't)
    N_spatial = train_data.shape[-1]
    iface = _detect_model_interface(model, device, N=N_spatial, T_IN=T_IN)

    def forward(x_input):
        if iface == 'grid':
            B, N, _ = x_input.shape
            grid = torch.linspace(0, 1, N, device=device).view(1, N, 1).expand(B, N, 1)
            out = model(x_input, grid)
        else:
            out = model(x_input)
        if isinstance(out, tuple):
            out = out[0]
        # Normalize output shape to (B, N, 1)
        while out.dim() > 3 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        if out.dim() == 2:
            out = out.unsqueeze(-1)
        return out

    # ---- Phase 1: Teacher Forcing ----
    print("=== Phase 1: Teacher Forcing ===")
    optimizer = optim.AdamW(model.parameters(), lr=args.phase1_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.phase1_epochs)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6)
    criterion = nn.MSELoss()
    best_val = float('inf')
    best_state = None
    no_improve_count = 0
    patience = 12
    SCOUT_MODE = args.no_plateau  # skip plateau + early stopping

    for epoch in range(1, args.phase1_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            B, T, N = batch.shape
            if T <= T_IN:
                continue
            t0 = random.randint(0, T - T_IN - 1)
            x = batch[:, t0:t0 + T_IN, :].permute(0, 2, 1)
            y = batch[:, t0 + T_IN, :].unsqueeze(-1)
            pred = forward(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * B
        scheduler.step()

        val_rel, s1, s2, s3 = validate(model, val_loader, device, T_IN=T_IN, forward_fn=forward)
        print(f"[Phase1 Epoch {epoch:3d}] Val Rel-MSE: {val_rel:.6f} | S1:{s1:.6f} S2:{s2:.6f} S3:{s3:.6f}")
        if not SCOUT_MODE:
            plateau_scheduler.step(val_rel)
        if val_rel < best_val * 0.995:  # meaningful improvement
            best_val = val_rel
            no_improve_count = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_rel': val_rel}, os.path.join(args.save_dir, 'best_phase1.pt'))
        elif not SCOUT_MODE:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Load best before Phase 2
    if os.path.exists(os.path.join(args.save_dir, 'best_phase1.pt')):
        ckpt = torch.load(os.path.join(args.save_dir, 'best_phase1.pt'), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    # ---- Phase 2: Progressive Rollout ----
    print("=== Phase 2: Progressive Rollout ===")
    optimizer = optim.AdamW(model.parameters(), lr=args.phase2_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)
    plateau_scheduler2 = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7)
    best_val = float('inf')
    no_improve_count = 0
    patience2 = 15
    SCOUT_MODE = args.no_plateau

    for epoch in range(1, args.phase2_epochs + 1):
        model.train()
        total_loss = 0.0
        if epoch < 10:
            roll_steps = 1
        elif epoch < 50:
            roll_steps = int(1 + (epoch - 10) * (39 / 40))
        elif epoch < 100:
            roll_steps = int(40 + (epoch - 50) * (60 / 50))
        else:
            roll_steps = 100

        for batch in train_loader:
            batch = batch.to(device)
            B, T, N = batch.shape
            if T <= T_IN + roll_steps:
                continue
            t0 = random.randint(0, T - T_IN - roll_steps)
            loss = 0.0
            for step in range(roll_steps):
                x = batch[:, t0+step:t0+step+T_IN, :].permute(0, 2, 1)
                y = batch[:, t0+step+T_IN, :].unsqueeze(-1)
                pred = forward(x)
                loss += criterion(pred, y)
            loss = loss / roll_steps
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * B
        scheduler.step()

        val_rel, s1, s2, s3 = validate(model, val_loader, device, T_IN=T_IN, forward_fn=forward)
        print(f"[Phase2 Epoch {epoch:3d}] roll={roll_steps:2d} | Val Rel-MSE: {val_rel:.6f} | S1:{s1:.6f} S2:{s2:.6f} S3:{s3:.6f}")
        if not SCOUT_MODE:
            plateau_scheduler2.step(val_rel)
        if val_rel < best_val * 0.995:
            best_val = val_rel
            no_improve_count = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_rel': val_rel}, os.path.join(args.save_dir, 'best_phase2.pt'))
        elif not SCOUT_MODE:
            no_improve_count += 1
            if no_improve_count >= patience2:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Final: load best phase2 if it exists
    best_p2_path = os.path.join(args.save_dir, 'best_phase2.pt')
    if os.path.exists(best_p2_path):
        ckpt = torch.load(best_p2_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    print("Training completed.")

if __name__ == '__main__':
    main()
