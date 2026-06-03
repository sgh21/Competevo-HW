#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_PATH="${1:-}"
INSTALL_PREFIX="${2:-${HOME}/conda-envs/EAI}"

if [[ -z "${ARCHIVE_PATH}" ]]; then
  echo "Usage: $0 <env-archive.tar.gz> [install-prefix]" >&2
  echo "Example: $0 /tmp/EAI-conda-env.tar.gz \$HOME/conda-envs/EAI" >&2
  exit 1
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "ERROR: archive does not exist: ${ARCHIVE_PATH}" >&2
  exit 1
fi

if [[ -e "${INSTALL_PREFIX}" ]]; then
  echo "ERROR: install prefix already exists: ${INSTALL_PREFIX}" >&2
  echo "Choose another path or remove the existing directory manually." >&2
  exit 1
fi

mkdir -p "${INSTALL_PREFIX}"

echo "Unpacking '${ARCHIVE_PATH}' to '${INSTALL_PREFIX}'..."
tar -xzf "${ARCHIVE_PATH}" -C "${INSTALL_PREFIX}"

echo "Fixing absolute prefixes with conda-unpack..."
"${INSTALL_PREFIX}/bin/conda-unpack"

echo
echo "Done."
echo "Activate with:"
echo "  source ${INSTALL_PREFIX}/bin/activate"
echo
echo "Quick check:"
echo "  python -V"
echo "  python -c \"import torch, gymnasium, mujoco; print('ok')\""
