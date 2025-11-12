#!/usr/bin/env bash
set -euo

# ------------------------------------------------------------------- #
# Pick a free port
# ------------------------------------------------------------------- #

if [[ -n "${BASH_VERSINFO:-}" ]] && (( BASH_VERSINFO[0] >= 4 )); then
    set -o pipefail
fi

pick_port() {
    while :; do
        port=$(shuf -n 1 -i 49152-65535)
        netstat -atun | grep -q "$port" || break
    done
    echo "$port"
}


# ------------------------------------------------------------------- #
# Args
# ------------------------------------------------------------------- #
TRAIN_CFG="$(realpath configs/train.json)"
EVAL_CFG="$(realpath configs/eval.json)"

COMMON_ARGS=(
    --train-config   "$TRAIN_CFG"
)

# ------------------------------------------------------------------ #
# 1. SLURM
# ------------------------------------------------------------------ #
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Running on SLURM (job ${SLURM_JOB_ID})"

    # SLURM environment
    export NCCL_DEBUG=INFO
    export NCCL_P2P_DISABLE=1
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_DISTRIBUTED_DEBUG=DETAIL

    # Master address / port
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    export MASTER_ADDR
    PORT=$(pick_port)
    echo "MASTER_ADDR=$MASTER_ADDR  PORT=$PORT"

    # torchrun (replaces torch.distributed.launch)
    torchrun \
        --nproc_per_node=4 \
        --nnodes="$SLURM_NNODES" \
        --rdzv_id="$SLURM_JOB_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${PORT}" \
        main.py \
        "${COMMON_ARGS[@]}" \
        --eval-config "$EVAL_CFG" \
        --run-eval

# ------------------------------------------------------------------ #
# 2. macOS
# ------------------------------------------------------------------ #
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Running locally on macOS (single GPU/CPU)"
    python main.py \
        "${COMMON_ARGS[@]}" \
        --eval-config "$EVAL_CFG" --run-eval   # uncomment if you want eval locally
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi