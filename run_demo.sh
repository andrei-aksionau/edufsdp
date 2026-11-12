#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# FSDP Inspection Launcher
#
# Usage:
#   ./run_demo.sh [--nprocs <num>]
#
# Notes:
#   - To disable boolean flags below, add a `no-` prefix
#     (e.g., --no-show-memory-training)
# ==============================================================================

# --- 1. Defaults & Argument Parsing ---
NPROCS=6

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nprocs)
            NPROCS="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1" >&2
            exit 1
            ;;
    esac
done

# --- 2. Environment Configurations ---
export OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore:Failed to initialize NumPy"

# --- 3. Execution ---
torchrun \
  --nproc-per-node="$NPROCS" \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  demo.py \
  --num-iterations 4 \
  --log-memory-transition \
  --log-memory-transition-first-iter-only \
  --no-log-checkpoint-memory-transition \
  2>&1 | sed '/Redirects are currently not supported in MacOs/d'
