#!/usr/bin/env bash
# The harness calls this to fetch OpenFHE for the reference submission.
# This submission does not use OpenFHE: it runs on Orion's Cheddar CUDA
# backend, which is built once from setup/SETUP.md for the machine's
# compute capability and is not fetched per run.
#
# The harness must not be modified (submission rules), so the no-op
# lives here, in scripts/, which submitters own.
echo "[get_openfhe] not used by this submission (Cheddar/CUDA backend)"
exit 0
