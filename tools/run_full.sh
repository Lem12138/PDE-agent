#!/usr/bin/env bash
# run_full.sh — 在 systemd --user 会话中启动完整工作流。
#
# 为什么需要它：直接在当前 shell 启动会继承受限的能力位（CapEff=0），
# CUDA 初始化失败、只能退回 CPU。由 systemd --user 直接 fork 的服务
# 拥有正常权限，可以使用 GPU。
#
# 用法：
#   systemd-run --user --unit=pde-captain --collect \
#       --working-directory="<仓库目录>" /bin/bash tools/run_full.sh
#
# 日志写到 output/run_<时间戳>.log，路径同时记在 output/run.logpath

set -u
cd "$(dirname "$0")/.." || exit 1

PY="/home/apde/miniconda3/bin/python3"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="output/run_${STAMP}.log"
mkdir -p output
echo "$LOG" > output/run.logpath

# ---- 启动前检查 GPU，不可用就不浪费几小时 ----
{
  echo "[launcher] $(date '+%F %T') 启动"
  echo "[launcher] 工作目录: $(pwd)"
  "$PY" - <<'PY'
import torch
ok = torch.cuda.is_available()
print("[launcher] cuda available:", ok)
if ok:
    print("[launcher] device:", torch.cuda.get_device_name(0))
    print("[launcher] cuda version:", torch.version.cuda)
PY
} > "$LOG" 2>&1

if ! grep -q "cuda available: True" "$LOG"; then
  echo "[launcher] GPU 不可用，放弃启动（不退回 CPU 空跑）" >> "$LOG"
  exit 1
fi

# ---- 启动主流程，输出续写到同一个日志 ----
exec "$PY" agent/captain.py \
  --data "/home/apde/pde_project/new_dataset/PDEBench/1D/diffusion-sorption/1D_diff-sorp_NA_NA.h5" \
  --max_rounds 5 \
  --num-workers 4 \
  --sigma 0.05 >> "$LOG" 2>&1
