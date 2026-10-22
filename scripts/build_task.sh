#!/usr/bin/env bash
# The harness calls this before every run. This submission ships built,
# so nothing is compiled here: the check is that the artifacts the
# harness will invoke are present, and that this machine's GPU can
# actually run them.
set -u

here="$(cd "$(dirname "$0")" && pwd)"
sub="$here/../submission"
missing=0

need() {
    if [ ! -e "$1" ]; then
        echo "[build_task] missing: ${1#$sub/}" >&2
        missing=1
    fi
}

# The compiled circuit and backend: one object each.
need "$sub/libs/fbs.cpython-310-x86_64-linux-gnu.so"
need "$sub/libs/orion.cpython-310-x86_64-linux-gnu.so"

# Data the compiled objects read from disk, by paths they derive from
# their own module locations.
need "$sub/data/boot_rotations.json"
for c in shallow_40_L6 shallow_40_L7 shallow_40_L8 shallow_40_L15 \
         shallow_40_L16 bootparam_40; do
    need "$sub/data/params32/$c.json"
done

# The CUDA extension, and the stages the harness runs as processes.
if ! ls "$sub"/libs/cheddar_native*.so >/dev/null 2>&1; then
    echo "[build_task] missing: the Cheddar CUDA extension" >&2
    missing=1
fi
for stage in server_encrypted_compute client_encode_encrypt_query \
             client_decrypt_decode client_postprocess; do
    need "$sub/build/$stage"
done
need "$sub/libs/libfbsnative.so"
need "$sub/libs/cuda-arch.txt"

if [ "$missing" = 1 ]; then
    cat >&2 <<'MSG'

  This submission is distributed built. The objects above are the
  deliverable, not build output, so a missing one means the archive is
  incomplete rather than that something needs compiling.

  It is built for Python 3.10 and for NVIDIA Blackwell (sm_120). See
  submission/README.md.
MSG
    exit 1
fi

# The extension holds one architecture's cubin plus its PTX. PTX is
# compiled forward to newer GPUs and never backward, so a card older
# than the build cannot run it -- and it does not say so: the kernels
# fail to load, nothing checks the launches, and the run completes with
# whatever the output buffers held, failing only at verify_result with a
# wrong answer. Refusing here is the difference between one clear line
# and debugging arithmetic.
built=$(tr -dc '0-9' < "$sub/libs/cuda-arch.txt")
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [ -z "$cap" ]; then
    echo "[build_task] no GPU visible; built for sm_$built" >&2
elif [ -z "$built" ]; then
    echo "[build_task] cannot read the built architecture; skipping the check" >&2
else
    have=$(( ${cap%%.*} * 10 + ${cap##*.} ))
    if [ "$have" -lt "$built" ]; then
        cat >&2 <<MSG

  [build_task] this GPU cannot run this submission.

    built for : sm_$built
    this GPU  : compute capability $cap (sm_$have)

  PTX is compiled forward, never backward, so there is no kernel for
  this card. Were this allowed to run, every stage would report
  completion and the result would simply be wrong.

  Build for this card, or run on compute capability ${built%?}.${built#??}
  or newer. See submission/README.md.
MSG
        exit 1
    fi
    [ "$have" -gt "$built" ] && echo "[build_task] sm_$built objects on sm_$have: the driver will JIT the PTX on first use" >&2
fi

echo "[build_task] submission artifacts present, built for sm_$built"
exit 0
