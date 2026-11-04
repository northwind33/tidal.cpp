# CUDA development container

The configuration in `.devcontainer/` runs the development tools on the GPU server.
Your local VS Code connects through Remote - SSH and Dev Containers. The default
CMake preset targets **Titan RTX / SM 7.5**, with CUDA 12.8.1 and Ubuntu 24.04.

## Host prerequisites

On the **GPU server**, install Docker Engine, an NVIDIA driver compatible with
CUDA 12.8, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The SSH user must be able to run Docker. The CUDA compiler comes from the container;
the server does not need a separate CUDA toolkit installation.

Verify driver access on the server and through Docker before opening VS Code:

```sh
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If the second command fails, complete NVIDIA's Docker runtime setup in the linked
installation guide. A successful host `nvidia-smi` alone does not establish container
GPU access. The container's startup script checks this again and fails explicitly
if CUDA or GPU access is missing.

## Open the repository

1. Install **Remote - SSH** and **Dev Containers** in local VS Code.
2. Connect to the Titan RTX server with **Remote-SSH: Connect to Host**.
3. Clone/open `northwind33/tidal.cpp` **on that server**, checking out the branch
   that contains `.devcontainer/` (initially `feat/tiny-hybrid-engine`).
4. Run **Dev Containers: Reopen in Container** from the Command Palette.

The default workspace mount uses the server's checkout, so edits and generated
files remain there. Do not open a laptop-only checkout with a remote Docker daemon
and expect it to be bind-mounted on the server. This workflow follows
[VS Code's remote-host instructions](https://code.visualstudio.com/remote/advancedcontainers/develop-remote-host).

The first image build installs development dependencies. The post-create script
checks `nvcc` and `nvidia-smi`, then runs `cmake --preset dev-cuda`, fetching the
pinned ggml dependency. It configures the project without starting the full build.
The normal development account is `vscode`, with its UID/GID mapped to the Linux
host user to avoid root-owned files in the checkout.

### Recover from a host-path CMake cache

If the checkout was configured on the host first, its cache may refer to
`/home/<user>/src/tidal.cpp` while the container sees `/workspaces/tidal.cpp`.
The post-create script detects a changed source or build path and moves the
complete `build/dev-cuda` directory to a unique `build/dev-cuda.stale.*` backup
before configuring again. This also isolates old dependency sub-build caches.
Existing output is preserved; a cache already configured at the current paths
is reused on subsequent runs.

If an older version of the script failed with this error, update the checkout
and rerun it **inside the existing container**; rebuilding the image is unnecessary:

```sh
git pull --ff-only origin feat/tiny-hybrid-engine
bash .devcontainer/post-create.sh
cmake --build --preset dev-cuda
```

## Build, test and run

Inside the container terminal:

```sh
cmake --build --preset dev-cuda
ctest --preset dev-cuda

# This explicitly exercises the GPU/CPU split, across every supported weight format.
python3 tests/test_reference.py --bin-dir build/dev-cuda/bin --backend cuda

mkdir -p models
build/dev-cuda/bin/tidal-dummy --output models/dev-q4.gguf --type q4_0
build/dev-cuda/bin/tidal-cli --model models/dev-q4.gguf --backend cuda \
  --tokens 1,7,23,9,42,5,6,2,77,13,84,11,90 --steps 8
```

CTest runs the CPU reference tests, even in a CUDA build. The separate Python
command is the GPU numerical/placement check. The generator refuses to overwrite
existing model files; reuse the generated file or choose a new filename on a repeat.

The preset uses Ninja and `RelWithDebInfo`, exports `compile_commands.json`, and
enables ccache for C/C++/CUDA compilation. `GGML_CUDA_FA=OFF` skips unused GPU
FlashAttention kernels: this engine computes main attention on the CPU. GDN and
QSA indexer kernels remain enabled. Build output goes under `build/dev-cuda`, which
is separate from the earlier `build-cuda` instructions and ignored by Git.

Build parallelism defaults to four jobs. On a machine with limited free RAM, use
`cmake --build --preset dev-cuda --parallel 2`. The C++ and CMake Tools extensions
are installed automatically. Select `dev-cuda` in **CMake: Select Configure Preset**
if prompted. GDB can also be used directly:

```sh
gdb --args build/dev-cuda/bin/tidal-cli --model models/dev-q4.gguf --backend cuda \
  --tokens 1,2,3 --steps 1
```

`SYS_PTRACE` and the relaxed container seccomp profile support GDB. The container
uses `--gpus=all`; for a server with several GPUs, change that argument to
`--gpus=device=0` (or the desired device UUID) and rebuild the container. It does
not run privileged, mount the Docker socket, or install a host GPU driver.

## Another GPU or local overrides

SM 7.5 is the checked-in default for the Titan RTX. To use another target without
changing the shared preset, create a git-ignored `CMakeUserPresets.json`:

```json
{
  "version": 5,
  "configurePresets": [{
    "name": "my-cuda",
    "inherits": "dev-cuda",
    "binaryDir": "${sourceDir}/build/my-cuda",
    "cacheVariables": { "CMAKE_CUDA_ARCHITECTURES": "native" }
  }],
  "buildPresets": [{ "name": "my-cuda", "configurePreset": "my-cuda" }],
  "testPresets": [{
    "name": "my-cuda",
    "configurePreset": "my-cuda",
    "output": { "outputOnFailure": true }
  }]
}
```

After the default post-create configuration, run `cmake --preset my-cuda` and
`cmake --build --preset my-cuda`. `native` targets the GPU(s) exposed to this
container; select a fixed compute capability instead for portable binaries.

## CI and verification limits

`python3 tests/test_devcontainer.py` uses real CMake projects to check fresh and
repeated setup, relocated source/build paths, preservation of dependency caches,
and an unchanged build when GPU preflight fails. Only `nvcc` and `nvidia-smi` are
stubbed for this regression test; it does not exercise CUDA or the engine.

CI builds this Dockerfile and checks the non-root account, CUDA compiler, CMake,
Ninja, GDB and NumPy without requesting a GPU. The separate CUDA compile job
disables ggml CUDA VMM **only in CI** because hosted runners lack `libcuda.so.1`.
The development preset uses ggml's normal VMM setting, with the real driver
libraries supplied by NVIDIA Container Toolkit at container startup.

Image/toolchain smoke checks and CUDA compilation do not prove GPU runtime
correctness. Run the GPU command above on the Titan RTX to validate that boundary.
