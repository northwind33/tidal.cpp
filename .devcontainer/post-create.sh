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

# A bind-mounted checkout may contain a host build with different absolute paths.
# Preserve the complete tree: dependency sub-builds also have their own caches.
build_dir="$repo_root/build/dev-cuda"
cache_file="$build_dir/CMakeCache.txt"
if [[ -f "$cache_file" ]]; then
    cached_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache_file")"
    cached_build="$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "$cache_file")"
    if [[ "$cached_source" != "$repo_root" || "$cached_build" != "$build_dir" ]]; then
        backup_dir="$(mktemp -d "${build_dir}.stale.XXXXXX")"
        mv -T -- "$build_dir" "$backup_dir"
        printf 'CMake cache paths changed; preserved the previous build at %s\n' "$backup_dir"
        printf 'Previous source: %s\nPrevious build: %s\n' "$cached_source" "$cached_build"
    fi
fi

# Configure once, leaving the potentially lengthy CUDA compilation to the user.
cmake --preset dev-cuda
printf '\nReady: cmake --build --preset dev-cuda\n'
printf 'CPU reference tests: ctest --preset dev-cuda\n'
printf 'GPU checks: python3 tests/test_reference.py --bin-dir build/dev-cuda/bin --backend cuda\n'
