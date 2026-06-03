#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-EAI}"
OUTPUT_PATH="${2:-/tmp/${ENV_NAME}-conda-env.tar.gz}"
STRICT_CONDA_PACK="${STRICT_CONDA_PACK:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found in PATH." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "ERROR: conda env '${ENV_NAME}' does not exist." >&2
  exit 1
fi

if ! conda run -n "${ENV_NAME}" python -c "import conda_pack" >/dev/null 2>&1; then
  echo "conda-pack is not installed in '${ENV_NAME}', installing it with pip..."
  conda run -n "${ENV_NAME}" python -m pip install conda-pack
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "Packing conda env '${ENV_NAME}' to '${OUTPUT_PATH}'..."
PACK_ARGS=(conda-pack -n "${ENV_NAME}" -o "${OUTPUT_PATH}" --force --n-threads -1)

if [[ "${STRICT_CONDA_PACK}" != "1" ]]; then
  echo "Using --ignore-missing-files to tolerate conda/pip metadata conflicts."
  PACK_ARGS+=(--ignore-missing-files)
fi

conda run -n "${ENV_NAME}" "${PACK_ARGS[@]}"

echo
echo "Done."
echo "Archive: ${OUTPUT_PATH}"
echo "Copy it to the target Ubuntu machine, then run scripts/unpack_conda_env.sh there."
