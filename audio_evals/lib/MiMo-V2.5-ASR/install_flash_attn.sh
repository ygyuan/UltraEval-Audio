#!/usr/bin/env bash
# ==============================================================================
#  install_flash_attn.sh
# ------------------------------------------------------------------------------
#  Helper to install ``flash_attn`` into the MiMo-V2.5-ASR isolated venv on a
#  host without public-internet access.
#
#  Strategy (in order):
#    1) Verify torch is present (the isolate framework should already have
#       installed it via asr_requirements.txt). If missing, install it from
#       the Tencent-Cloud mirror.
#    2) Try to *copy* an already-compiled ``flash_attn`` package from a
#       sibling conda env (default: ~/backup/miniconda3/envs/qwen3-tts).
#       This works when:
#         - the sibling env's Python minor version matches (cp312 == cp312)
#         - the sibling env's torch version + CUDA version match
#       This is the fastest path (no compilation, ~1 minute).
#    3) Otherwise, fall back to ``pip install --no-build-isolation
#       flash_attn==2.7.4.post1`` from the Tencent-Cloud PyPI mirror.
#       This compiles from source: requires nvcc + 30~60 min + plenty of RAM.
#
#  Usage:
#      bash audio_evals/lib/MiMo-V2.5-ASR/install_flash_attn.sh
# ==============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# Paths & configuration
# -----------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_DIR="${REPO_ROOT}/envs/MiMo-V2.5-ASR"
DONOR_ENV="${MIMO_FLASH_ATTN_DONOR:-/apdcephfs_qy3/share_301069248/users/yougenyuan/backup/miniconda3/envs/qwen3-tts}"
PIP_INDEX_URL="${ULTRAEVAL_PIP_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple/}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[ERROR] venv not found: ${VENV_DIR}" >&2
    echo "        Run the eval script first so that the isolate framework can create it." >&2
    exit 1
fi

VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

echo "[INFO] Target venv : ${VENV_DIR}"
echo "[INFO] Donor  env  : ${DONOR_ENV}"
echo "[INFO] PyPI mirror : ${PIP_INDEX_URL}"

# -----------------------------------------------------------------------------
# Step 0 -- detect Python minor version of the venv
# -----------------------------------------------------------------------------
PY_MM="$("${VENV_PY}" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "[INFO] venv python: ${PY_MM}"
SITE_PKG="${VENV_DIR}/lib/python${PY_MM}/site-packages"

# -----------------------------------------------------------------------------
# Step 1 -- ensure torch is installed in the venv
# -----------------------------------------------------------------------------
if ! "${VENV_PY}" -c 'import torch' >/dev/null 2>&1; then
    echo "[INFO] torch not found in venv, installing torch==2.6.0 ..."
    "${VENV_PIP}" install -i "${PIP_INDEX_URL}" torch==2.6.0 torchaudio==2.6.0
fi

VENV_TORCH_VER="$("${VENV_PY}" -c 'import torch;print(torch.__version__)')"
VENV_TORCH_CUDA="$("${VENV_PY}" -c 'import torch;print(torch.version.cuda)')"
echo "[INFO] venv torch  : ${VENV_TORCH_VER} (cuda=${VENV_TORCH_CUDA})"

# -----------------------------------------------------------------------------
# Step 2 -- short-circuit if flash_attn already importable
# -----------------------------------------------------------------------------
if "${VENV_PY}" -c 'from flash_attn import flash_attn_varlen_func' >/dev/null 2>&1; then
    FA_VER="$("${VENV_PY}" -c 'import flash_attn;print(flash_attn.__version__)')"
    echo "[OK] flash_attn ${FA_VER} is already installed and importable."
    exit 0
fi

# -----------------------------------------------------------------------------
# Step 3 -- try to clone flash_attn from the donor env
# -----------------------------------------------------------------------------
clone_from_donor() {
    if [[ ! -x "${DONOR_ENV}/bin/python" ]]; then
        echo "[WARN] donor env not found, skipping clone path."
        return 1
    fi

    local DONOR_PY_MM
    DONOR_PY_MM="$("${DONOR_ENV}/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
    if [[ "${DONOR_PY_MM}" != "${PY_MM}" ]]; then
        echo "[WARN] donor python (${DONOR_PY_MM}) != venv python (${PY_MM}), cannot clone."
        return 1
    fi

    local DONOR_TORCH_VER DONOR_TORCH_CUDA DONOR_FA_VER
    DONOR_TORCH_VER="$("${DONOR_ENV}/bin/python" -c 'import torch;print(torch.__version__)' 2>/dev/null || true)"
    DONOR_TORCH_CUDA="$("${DONOR_ENV}/bin/python" -c 'import torch;print(torch.version.cuda)' 2>/dev/null || true)"
    DONOR_FA_VER="$("${DONOR_ENV}/bin/python" -c 'import flash_attn;print(flash_attn.__version__)' 2>/dev/null || true)"
    echo "[INFO] donor torch : ${DONOR_TORCH_VER} (cuda=${DONOR_TORCH_CUDA})"
    echo "[INFO] donor fa    : ${DONOR_FA_VER}"

    if [[ -z "${DONOR_FA_VER}" ]]; then
        echo "[WARN] donor env has no flash_attn, cannot clone."
        return 1
    fi
    if [[ "${DONOR_TORCH_VER}" != "${VENV_TORCH_VER}" ]]; then
        echo "[WARN] donor torch != venv torch, ABI mismatch likely; skipping clone."
        return 1
    fi
    if [[ "${DONOR_TORCH_CUDA}" != "${VENV_TORCH_CUDA}" ]]; then
        echo "[WARN] donor cuda != venv cuda, ABI mismatch likely; skipping clone."
        return 1
    fi

    local DONOR_SITE
    DONOR_SITE="${DONOR_ENV}/lib/python${DONOR_PY_MM}/site-packages"
    echo "[INFO] copying flash_attn from ${DONOR_SITE} ..."
    cp -rL "${DONOR_SITE}/flash_attn" "${SITE_PKG}/"
    cp -rL "${DONOR_SITE}"/flash_attn-*.dist-info "${SITE_PKG}/" 2>/dev/null || true
    # Compiled CUDA extensions live at the top level of site-packages.
    for ext in "${DONOR_SITE}"/flash_attn_2_cuda*.so \
               "${DONOR_SITE}"/flash_attn_3_cuda*.so; do
        [[ -e "${ext}" ]] && cp -L "${ext}" "${SITE_PKG}/"
    done
    [[ -d "${DONOR_SITE}/flash_attn_3" ]] && cp -rL "${DONOR_SITE}/flash_attn_3" "${SITE_PKG}/"

    if "${VENV_PY}" -c 'from flash_attn import flash_attn_varlen_func' >/dev/null 2>&1; then
        echo "[OK] flash_attn cloned successfully from donor env."
        return 0
    fi
    echo "[WARN] clone path failed import check, falling back to source build."
    return 1
}

if clone_from_donor; then
    exit 0
fi

# -----------------------------------------------------------------------------
# Step 4 -- fall back to source build with --no-build-isolation
# -----------------------------------------------------------------------------
echo "[INFO] Falling back to source build (this can take 30~60 min)."
"${VENV_PIP}" install -i "${PIP_INDEX_URL}" ninja packaging wheel 'setuptools<81'
MAX_JOBS="${MAX_JOBS:-4}" "${VENV_PIP}" install \
    --no-build-isolation \
    -i "${PIP_INDEX_URL}" \
    flash_attn==2.7.4.post1

"${VENV_PY}" -c 'from flash_attn import flash_attn_varlen_func; import flash_attn; print("[OK] flash_attn", flash_attn.__version__)'
