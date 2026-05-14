#!/bin/bash
# rcp_publish.sh — upload rcp_v1 data for one or more models to HF, then delete .npy files locally.
# Keeps caz JSON files in place so the extraction skip-logic still works on re-run.
#
# Usage: rcp_publish.sh <model_id> [model_id2 ...]
#   e.g. rcp_publish.sh meta-llama/Llama-3.2-3B-Instruct EleutherAI/pythia-70m
#
# Slug derivation matches extract.py: replace / and - with _

HF_REPO="james-ra-henry/Rosetta-Activations"
RCP_ROOT="${HOME}/rosetta_data/rcp_v1"

for model_id in "$@"; do
    slug=$(echo "$model_id" | tr '/-' '_')
    local_dir="${RCP_ROOT}/${slug}"

    if [ ! -d "$local_dir" ]; then
        echo "[rcp_publish] WARNING: ${local_dir} not found — skipping ${model_id}"
        continue
    fi

    npy_count=$(find "$local_dir" -name "*.npy" | wc -l)
    if [ "$npy_count" -eq 0 ]; then
        echo "[rcp_publish] No .npy files in ${slug} — skipping upload (may already be published)"
        continue
    fi

    echo "[rcp_publish] Uploading ${slug} (${npy_count} .npy files) to HF rcp_v1/ ..."
    hf upload "$HF_REPO" "${local_dir}/" "rcp_v1/${slug}/" --repo-type dataset

    echo "[rcp_publish] Deleting .npy files for ${slug} ..."
    find "$local_dir" -name "*.npy" -delete
    echo "[rcp_publish] Done: ${slug}"
done
