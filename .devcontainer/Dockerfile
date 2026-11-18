FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        gdb \
        git \
        ninja-build \
        openssh-client \
        python3 \
        python3-numpy \
        python3-venv \
        sudo \
        tmux \
    && rm -rf /var/lib/apt/lists/*

# Login shells can replace the image's PATH. Keep nvcc available in VS Code's
# login-shell environment probe and in ordinary interactive terminals.
RUN printf 'export PATH="/usr/local/cuda/bin:$PATH"\n' > /etc/profile.d/tidal-cuda.sh

# Ubuntu 24.04 images may already reserve UID 1000 for 'ubuntu'. Reuse that
# account so Dev Containers can map its UID/GID to the Linux host's user.
RUN if getent passwd 1000 >/dev/null; then \
        existing_user="$(getent passwd 1000 | cut -d: -f1)"; \
        if [ "$existing_user" != vscode ]; then \
            usermod --login vscode --home /home/vscode --move-home "$existing_user"; \
        fi; \
    else \
        useradd --uid 1000 --user-group --create-home vscode; \
    fi \
    && usermod --shell /bin/bash vscode \
    && printf 'vscode ALL=(root) NOPASSWD:ALL\n' > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode \
    && install -d -o vscode -g "$(id -gn vscode)" \
        /home/vscode/.cache /home/vscode/.cache/ccache

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CCACHE_DIR=/home/vscode/.cache/ccache

USER vscode
WORKDIR /home/vscode
