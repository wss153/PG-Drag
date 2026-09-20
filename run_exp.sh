#!/bin/bash
# HAPAP Deformation Pipeline
# Features:
#   - Stage 1: Keypoint + SDS (with ROI masking) + Smooth-ARAP
#   - Stage 2: SDS + Handle Freeze + SSP (Sparse Shape Prior)

export JAXTYPING_TYPECHECKER=beartype
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline

# SSP is enabled by default in config (ssp.enabled=True)
# Override SSP parameters via environment variables if needed:
# export APAP_SSP_ALPHA=1.0     # L_surf weight
# export APAP_SSP_BETA=0.3      # L_io weight
# export APAP_SSP_M=4096        # Surface samples
# export APAP_SSP_K=4096        # Volume samples
# export APAP_SSP_LPEAK=0.10    # Peak lambda
# export APAP_SSP_LTAIL=0.02    # Tail lambda

PYTHONPATH=. python scripts/exp/batch/batch_deform_meshes.py \
    --data-list-path configs/deform_meshes/data/apap_3d.txt \
    --out-root outputs/apap-3d-ssp-$(date +%Y%m%d_%H%M%S) \
    --gpu-ids 1
