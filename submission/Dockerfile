# Environment for the pre-compiled submission.
#
# This submission ships built: the Python is compiled to extension
# modules and the CUDA work is a native extension with Cheddar linked
# into it. Compiled objects carry an ABI, so the environment is not a
# convenience here, it is part of the artifact. The three pins that
# matter:
#
#   Python 3.10   the objects are cpython-310-x86_64-linux-gnu; a 3.11
#                 or 3.12 interpreter will not import them. Ubuntu
#                 22.04 provides 3.10 as its default python3.
#   libcudart.so.12   the extension links the CUDA 12 runtime.
#   libgmp.so.10      Cheddar's only other non-system dependency; it is
#                 linked shared because the distro's static libgmp is
#                 built without -fPIC.
#
# Everything else the extension needs is inside it. `ldd` on
# submission/libs reports nothing beyond those two and libc.
#
# Build from the repository root, not from submission/, because the
# context has to include harness/. The -t name is local and arbitrary:
#
#   docker build -f submission/Dockerfile -t fbs-submission .
#
# Run a tier. The command is the same one you would run outside the
# container, spelled out in full, with the GPU and a writable mount for
# datasets and results (the harness generates each dataset on first use,
# which is slow enough to be worth keeping between runs):
#
#   docker run --rm --gpus all --user "$(id -u):$(id -g)" \
#       -v "$PWD/datasets:/fbs/datasets" \
#       -v "$PWD/io:/fbs/io" \
#       -v "$PWD/measurements:/fbs/measurements" \
#       fbs-submission \
#       python3 harness/run_submission.py <tier> --num_runs <runs>
#
# --user keeps the output yours. There is no USER below, so without it
# the container runs as root: the run still completes, but what it
# writes into the mounted directories comes out owned by root on the
# host. Nothing here needs root -- as the invoking user, /fbs is
# read-only and every write goes to a mount.
#
# The three mounted directories are tracked in the repository with a
# .gitkeep each, so they exist in any clone. Docker creates a missing
# bind-mount source itself, as root, and --user cannot affect that,
# so one that already exists cannot be created by the wrong user.
#
# Sizes are 0 toy, 1 small, 2 medium, 3 large; add --count_only for the
# Count interface. There is deliberately no ENTRYPOINT: the invocation
# should read as the specification it is, not as arguments appended to
# something the image decided. See submission/README.md for what the submission does
# and for the GPU architecture it is built against, which this image
# cannot change.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# With --user the caller cannot write /fbs, so Python would try to cache
# bytecode beside the stage shims and silently fail on every run. It
# also keeps .pyc files out of a submission whose point is that the
# Python is compiled.
ENV PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libgmp10 \
        coreutils \
    && rm -rf /var/lib/apt/lists/*

# Cheddar bundles RMM, which opens the CUDA runtime by name while
# running instead of linking against it. It tries exactly two names,
# "libcudart.so.<major>.0" and then "libcudart.so", and this image has
# neither: it ships libcudart.so.12 and libcudart.so.12.8.90, because
# the unversioned symlink belongs to the development package. Without
# the link below, every GPU stage dies at setup with "Unable to dlopen
# cudart" -- after the client stages have run and look fine, which is
# what makes it easy to misread. One symlink is cheaper than pulling in
# cuda-cudart-dev for its headers.
RUN ln -s libcudart.so.12 /usr/local/cuda/lib64/libcudart.so

WORKDIR /fbs

# Upstream's requirements file, unmodified: numpy only. Built and tested
# against 2.2.6, but the bound is left as upstream wrote it because the
# shipped objects do not link numpy's C API -- buffers cross through
# pybind11, not through numpy's ABI.
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY harness/ harness/
COPY scripts/ scripts/
# Carries the notices too: submission/LICENSE-3RD-PARTIES.md and
# submission/licenses/ sit beside the objects they cover, and the image
# redistributes the same binaries.
COPY submission/ submission/
COPY README.md LICENSE.md ./

# No ENTRYPOINT, so the harness invocation appears in full on the
# command line. A bare `docker run -it` gives a shell instead.
# build_task.sh only verifies that the artifacts are present; nothing is
# compiled here.
CMD ["bash"]
