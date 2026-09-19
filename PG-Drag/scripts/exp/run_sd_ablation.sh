#!/usr/bin/env bash
# PG-Drag (HAPAP) Symmetric Dirichlet ablation vs SSP
# Stage1 SDS=0, Stage2=600 steps, 3 representative cases

set -euo pipefail
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate YAPAP
cd "$(dirname "$0")/../.."
export JAXTYPING_TYPECHECKER=beartype WANDB_MODE=offline PYTHONPATH=. HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="outputs/sd_ablation/${STAMP}"
LOGDIR="outputs/batch_logs"
mkdir -p "$ROOT" "$LOGDIR"
MASTER="${LOGDIR}/sd_ablation_${STAMP}.log"

COMMON=(
  pipe-cfg:stable-diffusion3-d-pipe-config
  --pipe-cfg.lambda-sds-stage1 0.0
  --pipe-cfg.stage2-steps 600
  --pipe-cfg.no-virtual-seams
)

declare -a CASES=(
  "fox_doll|000|fox_doll_0000"
  "blue_wolf|004|blue_wolf_0000"
  "moai_statue|000|moai_statue_0000"
)

run_one() {
  local name=$1 kp=$2 lora=$3 variant=$4 gpu=$5
  local extra_flags=("${@:6}")
  local out="${ROOT}/${variant}/${name}_handle-${kp}_anchor-${kp}"
  local log="${LOGDIR}/sd_ablation_${variant}_${name}_${STAMP}.log"
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] START ${variant}/${name} gpu=${gpu}" | tee -a "$MASTER" "$log"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/exp/deform_meshes.py \
    --mesh-file "data/apap_3d/processed/${name}/mesh.obj" \
    --handle-file "data/apap_3d/processed/${name}/keypoints/${kp}/user_single_keypoints.txt" \
    --anchor-file "data/apap_3d/processed/${name}/keypoints/${kp}/constraint_single_keypoints.txt" \
    --out-dir "$out" \
    --lora-dir "data/lora_ckpts/apap_3d/legacy/${lora}" \
    "${COMMON[@]}" \
    "${extra_flags[@]}" \
    >> "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE ${variant}/${name} exit=$?" | tee -a "$MASTER" "$log"
}

echo "ROOT=$ROOT" | tee "$MASTER"

# Wave 1: baseline SSP (3 cases parallel on 2 GPUs - run 2+1)
for i in 0 1; do
  IFS='|' read -r name kp lora <<< "${CASES[$i]}"
  gpu=$((i % 2))
  run_one "$name" "$kp" "$lora" baseline_ssp "$gpu" &
done
wait
IFS='|' read -r name kp lora <<< "${CASES[2]}"
run_one "$name" "$kp" "$lora" baseline_ssp 0 &
wait

# SD weight sweep
for w in 0.001 0.01 0.1 1.0; do
  variant="sd_${w}"
  for i in "${!CASES[@]}"; do
    IFS='|' read -r name kp lora <<< "${CASES[$i]}"
    gpu=$((i % 2))
    run_one "$name" "$kp" "$lora" "$variant" "$gpu" \
      --pipe-cfg.ssp.no-enabled \
      --pipe-cfg.sd-reg.enabled \
      --pipe-cfg.sd-reg.weight "$w" &
    if (( i % 2 == 1 )); then wait; fi
  done
  wait
done

echo "ALL DONE ROOT=$ROOT" | tee -a "$MASTER"
ln -sfn "$(realpath "$ROOT")" outputs/sd_ablation/latest
