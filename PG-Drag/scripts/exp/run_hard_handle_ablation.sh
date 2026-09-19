#!/usr/bin/env bash
# Hard-Handle ablation: baseline (soft handle + freeze) vs hard Poisson handle
# Reads all cases from apap_3d.txt (JAPAP or HAPAP copy).

set -euo pipefail
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate YAPAP
cd "$(dirname "$0")/../.."
export JAXTYPING_TYPECHECKER=beartype WANDB_MODE=offline PYTHONPATH=. HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

APAP_3D_TXT="${APAP_3D_TXT:-../JAPAP/configs/deform_meshes/data/apap_3d.txt}"
NUM_GPUS="${NUM_GPUS:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="outputs/hard_handle_ablation/${STAMP}"
LOGDIR="outputs/batch_logs"
mkdir -p "$ROOT" "$LOGDIR"
MASTER="${LOGDIR}/hard_handle_ablation_${STAMP}.log"

COMMON=(
  pipe-cfg:stable-diffusion3-d-pipe-config
  --pipe-cfg.stage2-steps 600
)

declare -a CASES=()
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line// /}"
  [[ -z "$line" ]] && continue
  mesh_path="${line%%,*}"
  # name: .../processed/NAME/mesh.obj
  name=$(echo "$mesh_path" | sed -n 's|.*/processed/\([^/]*\)/mesh\.obj|\1|p')
  kp=$(echo "$line" | sed -n 's|.*keypoints/\([^/]*\)/user_single_keypoints\.txt.*|\1|p')
  lora=$(echo "$line" | awk -F', ' '{print $NF}' | sed 's|.*/||')
  [[ -n "$name" && -n "$kp" && -n "$lora" ]] || continue
  CASES+=("${name}|${kp}|${lora}")
done < "$APAP_3D_TXT"

echo "Parsed ${#CASES[@]} cases from ${APAP_3D_TXT}" | tee "$MASTER"
echo "ROOT=$ROOT NUM_GPUS=$NUM_GPUS SKIP_EXISTING=$SKIP_EXISTING" | tee -a "$MASTER"

run_one() {
  local name=$1 kp=$2 lora=$3 variant=$4 gpu=$5
  shift 5
  local extra_flags=("$@")
  local out="${ROOT}/${variant}/${name}_handle-${kp}_anchor-${kp}"
  local log="${LOGDIR}/hard_handle_${variant}_${name}_${STAMP}.log"
  if [[ "$SKIP_EXISTING" == "1" && -f "${out}/metrics/eval_metrics.json" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP ${variant}/${name} (exists)" | tee -a "$MASTER"
    return 0
  fi
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

worker() {
  local gpu=$1
  local idx=$2
  local total=${#CASES[@]}
  while (( idx < total )); do
    IFS='|' read -r name kp lora <<< "${CASES[$idx]}"
    run_one "$name" "$kp" "$lora" baseline_soft "$gpu"
    run_one "$name" "$kp" "$lora" hard_handle "$gpu" --pipe-cfg.hard-handle
    idx=$((idx + NUM_GPUS))
  done
}

for ((g=0; g<NUM_GPUS; g++)); do
  worker "$g" "$g" &
done
wait

python scripts/eval/aggregate_hard_handle_ablation.py "$ROOT"

echo "ALL DONE ROOT=$ROOT (${#CASES[@]} cases × 2 variants)" | tee -a "$MASTER"
ln -sfn "$(realpath "$ROOT")" outputs/hard_handle_ablation/latest
