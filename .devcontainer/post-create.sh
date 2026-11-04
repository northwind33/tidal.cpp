#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v nvcc >/dev/null; then
    echo "CUDA compiler missing. Rebuild using the repository's CUDA devel image." >&2
    exit 1
fi
nvcc --version

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; then
    echo "GPU access is unavailable inside the container." >&2
    echo "Check the host NVIDIA driver and NVIDIA Container Toolkit; see docs/devcontainer.md." >&2
    exit 1
fi

# Configure once, leaving the potentially lengthy CUDA compilation to the user.
cmake --preset dev-cuda
printf '\nReady: cmake --build --preset dev-cuda\n'
printf 'CPU reference tests: ctest --preset dev-cuda\n'
printf 'GPU checks: python3 tests/test_reference.py --bin-dir build/dev-cuda/bin --backend cuda\n'
