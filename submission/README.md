# Fetch-by-Similarity — Count and Fetch, multi-GPU CKKS

NYU / LG Electronics USA, Inc.'s
submission to the [fetch-by-similarity workload](https://fhe-benchmarking.github.io/fetch-by-similarity/)
of the HomomorphicEncryption.org FHE benchmarking suite.

Encrypted similarity search over a database sharded across GPUs, on
[Orion](https://github.com/baahl-nyu/orion)'s
[Cheddar](https://github.com/scale-snu/cheddar-fhe) CUDA backend.

Both interfaces, all four instance sizes: eight configurations, every
one of which runs end to end and passes every query against the
harness's own cleartext oracle. This document records what the
implementation is, what it assumes, and what it measures.

## Overview

The workload is an encrypted nearest-neighbour search. A database of
unit-norm key vectors, each carrying a small payload, is encrypted by
its owner and handed to a server. The server is then asked about a
query vector it cannot read, and must answer without learning the
query, the records, or which records matched. A match is a cosine
similarity above 0.8, and the workload promises at most 32 of them per
query.

| Tier | Records | Dimension |
|---|---|---|
| toy | 1,000 | 128 |
| small | 50,000 | 128 |
| medium | 1,000,000 | 256 |
| large | 20,000,000 | 512 |

The workload defines two queries, and this submission implements both at
every size. **Count** returns how many records matched. **Fetch**
returns the matching records' payloads — seven small integers each —
without revealing which records they came from.

All computation is performed under CKKS on GPUs. The database is encoded so that
scoring the whole of it against a query is a matrix-vector product on
ciphertexts; a polynomial standing in for the step function turns scores
into match indicators; Count adds the indicators, and Fetch uses them to
gather payloads. The work is split across GPUs by database slice, one
process per card, and each size uses the fewest cards whose memory holds
its slice, so the smaller sizes run on one card and are directly
comparable with single-GPU submissions.

The server never holds a secret key, and the only value ever decrypted
is the answer: for Count a single ciphertext, not per-shard subtotals,
because subtotals would disclose how matches are distributed across the
database.

**This submission is distributed in compiled form.** The circuit, the
backend and the CUDA work are compiled objects under `submission/libs`,
the per-query stages are native programs under `submission/build`, and
nothing is compiled when the harness runs. `submission/Dockerfile`
defines the environment they were built against. The stage entry points
remain Python source, so the sequence of calls can be followed without
the implementation sources.

The remaining sections are organised as follows.

- [**Benchmark**](#benchmark) — the measurements, and the hardware they were
  obtained on.
- [**What is in `submission/`**](#what-is-in-submission) — every file the
  submission ships.
- [**Running it**](#running-it) — reproduction, the prerequisites and the
  container.
- [**Harness integration**](#harness-integration) — how the submission meets
  the suite's rules.
- [**How it works**](#how-it-works) — the circuit.
- [**Cryptographic parameters and
  security**](#cryptographic-parameters-and-security) — the parameter
  selection and its basis.
- [**Assumptions**](#assumptions-stated-explicitly) — what the
  implementation relies on the workload to guarantee.
- [**Status**](#status) — what runs, and the constraints that follow from
  the design.
- [**Licensing**](#licensing) — the terms covering this submission and the
  code compiled into it.
- [**Contact**](#contact) — the submitting groups, and who to ask.
- [**Appendix**](#appendix) — reference detail: the stage command interface,
  the client stages, the module layout and the environment variables.

## Benchmark

Collected 2026-10-09 on an AWS `g7e.48xlarge`: 8x NVIDIA RTX PRO 6000
Blackwell, 192 vCPU, with the objects in `submission/libs` exactly as
committed. Each figure is the mean of three runs of one invocation,
`--num_runs 3`. The columns are the ones the leaderboard shows by
default, in its order; `Srv` is its server-reported group.

**Count**

| Tier | GPUs | Keys | DB | Query | Total | Keygen | DB Enc | Q Enc | Compute | Srv Total | Srv Compute |
|---|---|---|---|---|---|---|---|---|---|---|---|
| toy | 1 | 416.0M | 320.0M | 1.3M | 2.9705s | 847.7ms | 351ms | 11.67ms | 13.23ms | 11.63ms | 7.097ms |
| small | 1 | 416.0M | 320.0M | 1.3M | 3.3994s | 852.6ms | 369.1ms | 11.77ms | 15.53ms | 13.86ms | 9.308ms |
| medium | 1 | 325.0M | 9.0G | 1.1M | 17.158s | 709.4ms | 4.5926s | 10.8ms | 83ms | 81.27ms | 76.75ms |
| large | 6 | 362.0M | 382.5G | 1.3M | 7.6565m | 757.1ms | 3.035m | 11.83ms | 782.7ms | 780.9ms | 774.6ms |

**Fetch**

| Tier | GPUs | Keys | DB | Query | Total | Keygen | DB Enc | Q Enc | Compute | Srv Total | Srv Compute |
|---|---|---|---|---|---|---|---|---|---|---|---|
| toy | 1 | 726.5M | 355.0M | 1.4M | 3.4966s | 1.1939s | 372ms | 12.93ms | 16.63ms | 14.93ms | 7.144ms |
| small | 1 | 726.5M | 358.0M | 1.4M | 3.846s | 1.22s | 420.7ms | 12.7ms | 18.43ms | 16.77ms | 9.295ms |
| medium | 1 | 344.0M | 10.1G | 1.3M | 17.984s | 746.6ms | 4.5653s | 12.03ms | 167.3ms | 165.6ms | 99.75ms |
| large | 8 | 4.8G | 388.5G | 5.0M | 8.5274m | 4.5787s | 3.149m | 39.23ms | 3.3497s | 3.3479s | 3.3417s |

The number of GPUs is derived rather than configured, which is why it
varies by row. Each tier uses the fewest cards that hold its database
resident: toy, small and medium each fit on one card of this machine, so
those rows are directly comparable to a single-GPU submission. Count
large takes six, and Fetch large all eight, because the payload stream
and the bootstrap the compaction requires raise what must remain
resident. No count is selected manually; each is derived from the memory
the device reports.

Two consequences follow for the interpretation of any figure reported
here: a figure is meaningful only from a run that passed, and the basis
for trusting a passing run is the oracle rather than the absence of an
error message.

The verification result requires care in interpretation. The harness
selects a random query unless `--seed` is given, and at toy a random
query frequently matches nothing; the run then reports
`PASS (All 0 payload vectors match)`, which verifies the plumbing but
none of the arithmetic. Seeds 2, 5 and 6 produce queries with matches at
every tier, and a meaningful PASS reports a non-zero count.

## What is in `submission/`

Every file the submission ships, and nothing is generated at run time.

```
submission/
├── README.md                                             # this document
├── LICENSE.md                                            # terms for this submission's own code
├── LICENSE-3RD-PARTIES.md                                # the third-party notices
├── licenses/                                             # their full texts, one file per project
├── Dockerfile                                            # the environment they were built in
│
├── client_preprocess_dataset.py                          # the five stages the harness runs as
├── client_key_generation.py                              #   Python; each is a thin entry point
├── client_encode_encrypt_db.py                           #   into the compiled circuit, not an
├── client_preprocess_query.py                            #   implementation. The last also starts
├── server_preprocess_dataset.py                          #   the shard pool and leaves it parked
├── server_shard_main.py                                  # one shard, spawned by the pool
├── server_watchdog_main.py                               # stops the pool when the harness exits
│
├── build/                                                # the four stages run once per query, as
│   ├── server_encrypted_compute                          #   native programs: the harness times
│   ├── client_encode_encrypt_query                       #   each as a whole process, so an
│   ├── client_decrypt_decode                             #   interpreter's start would be charged
│   └── client_postprocess                                #   to the measurement
│
├── libs/                                                 # the compiled objects
│   ├── fbs.cpython-310-x86_64-linux-gnu.so               # the circuit
│   ├── orion.cpython-310-x86_64-linux-gnu.so             # the backend
│   ├── cheddar_native32.cpython-310-x86_64-linux-gnu.so  # the CUDA extension
│   ├── libfbsnative.so                                   # hot loops, loaded with ctypes
│   └── cuda-arch.txt                                     # the architecture it was built for
│
└── data/                                                 # what the objects read from disk
    ├── params32/                                         # six prime chains, one per configuration
    └── boot_rotations.json                               # the bootstrap's rotation set
```

`datasets/`, `io/` and `measurements/` sit outside `submission/` and are
the harness's: the first two hold generated and intermediate data, the
third the result files. Each is tracked with a `.gitkeep` and is
otherwise written at run time.

## Running it

```bash
python3 harness/run_submission.py <tier> [--count_only] --num_runs <runs>
```

`<tier>` is 0 toy, 1 small, 2 medium, 3 large. `--count_only` selects
the Count interface and its absence selects Fetch. The harness writes
each run to `measurements/<instance>/results-<N>.json`, creating the
directory itself.

### Requirements

**This submission is distributed built.** Nothing is compiled when the
harness runs. The Python is compiled to extension modules, the CUDA
extension carries Cheddar linked statically, and the stages the harness
invokes as processes are binaries under `submission/build`. The sources
are not included.

The prerequisites are therefore environmental rather than a build
toolchain:

| Component | Requirement |
|---|---|
| GPU | NVIDIA Blackwell, compute capability 12.0. The extension ships an `sm_120` cubin and `compute_120` PTX, so it runs on Blackwell natively and on anything newer by JIT. PTX never carries backward, so an older card cannot run it. Check with `cuobjdump --list-elf submission/libs/cheddar_native32*.so` against `nvidia-smi --query-gpu=compute_cap` |
| Python | 3.10. Extension modules are tied to a minor version, and these are `cpython-310-x86_64-linux-gnu` |
| System | `libgmp.so.10` and a CUDA 12.8 runtime. The runtime has to be reachable under the plain name `libcudart.so`, not only as `libcudart.so.12`: Cheddar bundles RMM, which opens the CUDA runtime by name while running instead of linking against it, so the unversioned symlink has to be there. CUDA's runtime packages ship only the versioned name, which is why `submission/Dockerfile` adds the link |

`submission/Dockerfile` defines that environment, which is why it is
shipped: a compiled artifact is reproducible only together with the
environment it was built against. It pins Ubuntu 22.04 for Python 3.10,
installs `libgmp10`,
takes the CUDA 12.8 runtime from the base image, and adds the
`libcudart.so` symlink RMM needs. Build from the repository root, since
the context has to include `harness/`:

**A mismatch is rejected before the run begins.**
`scripts/build_task.sh`, which the harness runs before every run,
compares `nvidia-smi`'s compute capability against
`submission/libs/cuda-arch.txt`, written when the extension was built,
and stops with one line naming both. The check exists because a
mismatch is otherwise silent: the kernels cannot be loaded, nothing
tests the failed launches, and the run completes with a wrong answer. A
newer card is accepted, with a note that the driver will compile the PTX
on first use. Where no GPU is visible the check reports that and does
not block the run.

### Running in a container

```bash
docker build -f submission/Dockerfile -t fbs-submission .

docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v "$PWD/datasets:/fbs/datasets" \
    -v "$PWD/io:/fbs/io" \
    -v "$PWD/measurements:/fbs/measurements" \
    fbs-submission \
    python3 harness/run_submission.py <tier> [--count_only] --num_runs <runs>
```

The final line is the same invocation used outside the container. The
image declares no `ENTRYPOINT`, so the command being executed appears in
full rather than being supplied by the image. `fbs-submission` is the
local image tag assigned by `-t`. Invoked without a command,
`docker run --rm -it --gpus all fbs-submission` yields a shell.

`--user` preserves ownership of the output. The image declares no
`USER`, so without it the container runs as root: the run still
completes, but the datasets, results and logs it writes are owned by
root on the host and require `sudo` to remove. No part of the submission
requires root; under the invoking user `/fbs` is read-only and every
write is directed to a mount.

The three mounted directories are tracked in the repository, each with a
`.gitkeep`, so a clone already has them and they keep their ownership
either way. That is deliberate: Docker creates a missing bind-mount
source itself, as root, before the container starts, and `--user` cannot
affect that because it is not the container doing it.

The mounts are only to keep generated datasets and written results
outside the container; nothing requires them. All six recorded
configurations pass inside the image, with `Encrypted computation`
matching a run on the host.

### Large instances: data on another volume

Large generates about 41 GB of dataset and 390 GiB of ciphertext, which
is more than a root volume usually has. Nothing requires those to sit
next to the repository: they may be mounted from wherever the space is,
retaining only `measurements/` locally, as it holds the result files
that are committed.

```bash
mkdir -p <path/to/big/datasets> <path/to/big/io>

docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v <path/to/big/datasets>:/fbs/datasets \
    -v <path/to/big/io>:/fbs/io \
    -v "$PWD/measurements:/fbs/measurements" \
    fbs-submission \
    python3 harness/run_submission.py <tier> [--count_only] --num_runs <runs>
```

Mounting is preferred to symbolic links and to bind mounts, neither of
which is required. The harness resolves all paths from the working
directory, which is `/fbs` inside the container, so each directory may
be supplied from any host path. A symbolic link at `io/<tier>` would not
survive in any case: `run_submission.py` removes that directory before
each invocation, which deletes the link rather than its contents.

On an ephemeral volume, `measurements/` should be copied or committed
before the instance is stopped.

## Harness integration

`scripts/get_openfhe.sh` is a no-op: this submission does not use
OpenFHE. `scripts/build_task.sh` verifies that the shipped artifacts are
present and fails with a list if any is missing, since a missing object
means an incomplete archive rather than something to compile. The
`harness/` directory is upstream and unmodified.

**The stages that run once per query are compiled.** The harness times
every stage as a whole process (`run_submission.py`), so an
interpreter's start and exit are included in the figures the leaderboard
reports.

For each stage the harness runs `submission/<stage>.py` if that file is
present, and `submission/build/<stage>` otherwise
(`harness/utils.py:145`). The four below are therefore shipped without a
`.py`:

| Stage | Description |
|---|---|
| `server_encrypted_compute` | passes two file paths to the running shards and reads a number back; no key material, no cryptography |
| `client_encode_encrypt_query` | encodes and encrypts the query, in the 2^14 subring |
| `client_decrypt_decode` | decrypts the result, every ciphertext on its own thread |
| `client_postprocess` | writes `results.bin` |

Their command-line interface, and what they do and do not read from
their environment, are in the *Appendix*.

## How it works

The database is a matrix of unit-norm records; a query is a unit vector;
a match is a cosine similarity above 0.8. The computation is a
matrix-vector product under encryption, followed by a threshold — and
then either a sum, for Count, or a gather of the matching records'
payloads, for Fetch. Everything below the threshold is shared by both
interfaces.

**Diagonal encoding.** The database is split into `dim × dim` blocks,
each stored by its diagonals, so a matrix-vector product becomes
rotate-and-multiply: rotate the query `dim-1` ways, multiply each
rotation against the matching diagonal, accumulate.

**Complex-lane packing.** Two block-groups ride in one ciphertext, one
in the real lane and one in the imaginary, pre-scaled by 0.5. A single
ciphertext-ciphertext multiply covers both; the server separates them
with one conjugation. Halves the database size and the multiply count.

**Pre-rotated BSGS.** The baby-step/giant-step decomposition's giant
steps are baked into the plaintext database at enrolment as an index
roll, so only the baby-step rotations happen at query time. This costs
no VRAM and shrinks the rotation key set from `dim-1` keys to
`log2(n1) + dim/n1` — 4,772 MB to 308 MB, which accounts for the key
bandwidth reported above.

The baby step `n1` is chosen per instance size, since it is a function
of the record dimension: it must leave a non-trivial split, and at dim 128 a
baby step of 128 empties the giant set entirely, making all `dim-1`
key-switches baby steps and the decomposition a no-op. Toy and Small
use `n1 = 16`, which costs four rotation keys not already in the
EvalSum ladder (326 → 398 MB) and halves the query.

**Threshold.** A degree-15 or degree-31 Chebyshev polynomial standing in
for the indicator, fitted to the workload's promised score gap. See
*Assumptions*.

**Sharding.** The database is split by block-group across GPUs, one
process per device. Each shard replicates the rotated query set and
scores its own slice; one shard then adds the partial counts
**homomorphically** and returns a single ciphertext. Summing decrypted
per-shard counts on the client would disclose how matches are
distributed across the database, so the addition happens where there is
no secret key and only the total is ever decrypted.

**The database is transmitted with its public polynomial as a seed.** A
secret-key ciphertext is `(b, a)` with `a` uniform and independent of the
message, so `a` is sent as a 32-byte seed and regenerated by the shard —
half the bytes of every database and payload ciphertext. The generator is
SHAKE-128 with rejection sampling, so each coefficient is exactly uniform
below its prime; the C, CUDA and numpy implementations agree bit for bit
and are tested against each other. At Large the shard does not even
expand it: the seed stays on the device and `a` is regenerated inside the
multiply-accumulate, which is what allows that tier to fit.

**The query is transmitted in a 2¹⁴ subring.** At the shallow chains the query's
modulus fits the smaller ring's 128-bit bound, so it is encrypted under
its own small secret and the shard lifts it by an NTT index gather and
one key switch. This reduces query bandwidth fourfold at a cost of
approximately 1.5 ms. It is not available under a bootstrap; see
*Cryptographic parameters*.

### Fetch: returning the payloads

Two circuits, selected per instance size, because the appropriate
construction depends on the record count.

**Direct** (toy, small, medium). Each match's 63-bit payload is carried
as four 16-bit limbs, and the shard returns the indicator times each limb
channel, one level past the comparator. The client divides each product
by the indicator, which makes the answer exact regardless of how far a
match's indicator sits from its nominal value. Pairs of score
ciphertexts share one output through the complex lanes. The result size
grows with the database, which is why this circuit is used only through
Medium.

**Compact** (large). A fixed five ciphertexts, independent of the record
count:
a presence map and four payload-channel pairs. Every match is given its
rank within its slot column by masked suffix sums, a degree-31 equality
polynomial selects rank *r*, and the payloads are gathered and summed
onto one output row. Because the output size is fixed, **the server
learns nothing about how many records matched** — the response has the
same shape for zero matches as for thousands. It costs nine levels
past the comparator; at Large those come from a bootstrap rather than a
longer chain.

### Client and server are separate processes

Only the client holds a secret key. A shard holds evaluation keys and
ciphertexts: it cannot encrypt, cannot decrypt, and cannot derive a
rotation key it was not given: the backend refuses all three, and a
shard cannot verify its own arithmetic.

The client runs on the **CPU**. Its reference implementation is numpy
(`orion/backend/cheddar/cpu/`), checked against the CUDA implementation
in both directions; what the stages run is compiled and checked against
that: the per-query stages -- query encryption, decryption,
postprocessing -- are C programs checked against that reference, and key
generation and database encryption call a C library for their
arithmetic, likewise checked against it. It opens no CUDA context, so
every card belongs to a shard.

The shard pool is started once by `server_preprocess_dataset` and parked
on a request FIFO, because the harness runs every stage as a separate
process and a server that loaded its keys and database per query would
pay that inside the stage reported as `Compute`. A detached watchdog
stops the pool when the harness exits; the harness itself is unmodified.
A shard imports the Cheddar backend alone -- not torch or Orion's
framework -- maps its database file, regenerates the seeded `a` of every
ciphertext in the compiled library and imports them from a pinned buffer
in place; it uses CUDA managed memory only when the sizing says its slice
would not fit the card.

## Cryptographic parameters and security

CKKS, in Cheddar's 32-bit-word build, at ring degree `N = 2^16` (32,768
slots) and scale `2^40` throughout, with pregenerated rational-rescaling
prime chains. The chain is chosen by circuit depth, so it varies by
interface and size:

| Configuration | Dim | Comparator | Fetch circuit | Chain | Enc level | log Q | log P | **log(QP)** |
|---|---|---|---|---|---|---|---|---|
| Count toy / small | 128 | gap6 | — | `shallow_40_L7` | 6 | 340 | 186 | **526** |
| Count medium | 256 | gap5 | — | `shallow_40_L6` | 5 | 310 | 186 | **496** |
| Count large | 512 | gap6 | — | `shallow_40_L7` | 6 | 340 | 186 | **526** |
| Fetch toy / small | 128 | gap6 | direct | `shallow_40_L8` | 7 | 430 | 186 | **616** |
| Fetch medium | 256 | gap5 | direct | `shallow_40_L7` | 6 | 340 | 186 | **526** |
| Fetch large | 512 | gap6 | compact | `bootparam_40` | 16 | 1373 | 372 | **1745** |

`log(QP)` is the total modulus a lattice attack sees: the widest
ciphertext modulus `Q` the chain can reach, together with the
key-switching modulus `P`. The figures are computed from the chain files
through the implementation's own `level_np`/`evk_np` rather than read
off the prime lists, so this arithmetic is not duplicated.

**Seven of the eight configurations have a very large margin.** For a
ternary secret at `N = 2^16`, the
[HE security guidelines](https://eprint.iacr.org/2024/463) (Bossuat et
al., *Security Guidelines for Implementing Homomorphic Encryption*) put
the 128-bit classical boundary at roughly 1,750 bits of `log(QP)`. At
496-616 bits these chains use about a third of the budget, and security
at fixed `N` increases as `log(QP)` falls, so a shorter chain is strictly
more secure. They are short because the comparator is shallow and
nothing bootstraps.

**Fetch Large is the exception, and its margin is thin.** Bootstrapping
needs the depth, and `bootparam_40.json` is **upstream Cheddar's own
bootstrapping parameter set, used unmodified** — it is byte-identical to
`parameters/bootparam_40.json` in the Cheddar tree, and every Cheddar
boot chain at this ring is in the same band (`bootparam_30` 1768,
`bootparam_35` 1771, `bootparam_40` 1745). At **1745 against roughly
1,750** this is inside the boundary as the guidelines state it, but with
almost none of the margin the other rows have, and no independent
lattice-estimator analysis has been performed against it. A reviewer
requiring a larger margin at this configuration can select the
alternative described below. The figure is reported as measured rather
than characterised as comfortable.

**The conservative configuration is one environment variable.**
`FBS_BOOT_FETCH=0` runs Fetch Large on `shallow_40_L16` instead, at
**886 bits**, which puts every configuration in this submission under 900
bits with no bootstrapping anywhere. Both are measured on the same
machine:

| Fetch Large | `FBS_BOOT_FETCH=0` (L16) | Default (bootstrapped) |
|---|---|---|
| **log(QP)** | **886** | 1745 |
| bootstrapping | none | one per score ciphertext |
| `Compute` | 6.44 / 6.20 / 6.50 s | **3.206 / 3.252 / 3.250 s** |
| device slice | 126.0 GiB, 1.32x a card — pages | **80,822 MiB, 0.83x** |
| `DB` | 844.5G | **388.5G** |
| `Keys` | **1.1G** | 4.8G |

The default is the bootstrapped one because it is the configuration that
holds resident on the eight cards Fetch Large uses; the modulus and the 4.8 G key set are what
it costs. Nothing else in the submission is affected either way — Count
Large is on `shallow_40_L7` at 526 bits.

**The sparse secret, precisely.** The chain files carry a
`sparse_hamming_weight` of 32 for sparse-secret encapsulation, and
because the flag is derived as `sparse_hamming_weight > 0` it is **on in
every chain, not only the bootstrapping one**. So a sparse secret *is*
sampled and both switching keys *are* published, in all eight
configurations. What makes this sound is where each key lives:

- the key encrypted **under the sparse secret** (dense-to-sparse) is
  built at `dts_np`, which is **112 bits** of `log(QP)` — at `N = 2^16`
  that is a very hard lattice problem even at `h = 32`;
- the key at the **full** modulus (sparse-to-dense) is encrypted under
  the *dense* secret, so its hardness is the dense secret's, exactly like
  every other evaluation key.

The sparse secret is therefore never exposed at a large modulus. Earlier
revisions of this document stated that no sparse secret was sampled,
which was incorrect; the argument above supersedes it. The property is
enforced by a test, so that a chain raising the dense-to-sparse key's
modulus would fail that test rather than leave this paragraph
inaccurate.

**The client is the only holder of a secret key.** It is written to
client-private storage (mode 0600 in a 0700 directory) that no server
stage is given a path to, and removed when the run ends. It is the key
itself rather than a seed it could be regenerated from: storing a seed
would keep something equally sensitive on disk while also making the
error sampling reproducible. Seeds for the public polynomial `a` come
from `os.urandom` per ciphertext and never from the client's own
generator, which can be seeded for debugging and would then repeat `a` —
two ciphertexts sharing `a` can be subtracted apart without the secret.

## Assumptions, stated explicitly

The suite requires pre- and post-processing outside the encrypted
computation to be documented. Four aspects of this implementation
qualify.

**1. The comparator assumes the workload's promised score gap.** The
threshold polynomial is not accurate everywhere; it is accurate outside
an interval that the workload guarantees no score falls inside. Measured
against the official generator over 200 queries per size, the empirical
gap is *wider* than assumed at every dimension, so the assumption holds
with margin:

| Dimension | Assumed gap | Measured gap |
|---|---|---|
| dim 128 | [0.703, 0.845] | [0.433, 0.890] |
| dim 256 | [0.538, 0.866] | [0.337, 0.900] |

dim 512 uses the dim-256 gap, which is conservative: the gap scales as
`1/sqrt(dim)`, so it is wider at 512, not narrower.

**Degree is chosen per size**, and this is a correctness constraint
rather than a tuning parameter. The degree-15 fit miscounts at dim 128 —
5 of 200 queries wrong against the official generator — because the gap
there is 2.3× narrower and a steeper crossing oscillates. Its residuals also
accumulate with `K`, reaching the rounding budget at Large. So Medium
uses degree 15 and every other size uses degree 31. The submission
refuses the unsafe combination rather than allowing it.

**2. The database packing is plaintext preprocessing.** Normalization,
diagonalization and the BSGS pre-rotation all happen on the client
before encryption. They are index permutations and a scaling; they
compute no part of the similarity search.

**3. The compact Fetch circuit recovers at most 8 matches per slot
column.** It ranks matches within each of 4,096 columns and returns ranks
1-8, so the recoverable ceiling is 8 x 4,096 = 32,768 matches per score
ciphertext. This is a property of the output format, not of the search:
every match is still found and counted, and `--count_only` is unaffected.
The figure is `FETCH_RANKS` in `fbs/common/presets.py`, and the choice of
4,096 columns is what makes the column 8 rows tall — 244 records per
column at Medium rather than 1,953, which doubled the noise budget at the
gather. It applies to Large alone; toy,
small and medium use the direct circuit, which has no such ceiling.

**4. Payloads are integers, and are rounded on return.** The workload's
payloads are 7 channels of integers in `[0, 512)`
(`harness/generate_dataset.py`). Both circuits carry them through CKKS,
which is approximate, and the client rounds to the nearest integer; the
direct circuit additionally divides each product by the indicator so the
answer is exact however far the indicator sits from its nominal value.
Rounding is what allows an approximate scheme to return exact payloads,
and it is sound only because the payloads are integral; a real-valued
payload would require its own precision argument.

Everything else — the inner product, the threshold, the sum, the ranking
and the payload gather — is performed on ciphertexts.

## Status

**All eight configurations run end to end** — Count and Fetch at toy,
small, medium and large — and pass every query against the harness's own
cleartext oracle.

**Small cannot be sharded.** At 50,000 records the database is smaller
than a single block-group (`2 x slots = 65,536`), so `ceil(K / 65536) = 1`
and the run uses one GPU regardless of what the machine provides. The
multi-GPU design offers no benefit at that size; it targets databases
large enough to divide.

**Fetch Large requires eight cards**, with little margin; Count Large
holds on six. Its database is 156,672 ciphertexts, and it fits only
because the seed remains on the device and, for Fetch, the compaction's
levels come from a bootstrap rather than a longer chain — which is also
why Fetch requires the two cards Count does not. Count Large completes
in under a second and Fetch Large in a little over three. No other
submission has published at Large, including the reference, so those
rows have nothing to compare against.

**This submission is self-contained, and distributed as binaries.**
Orion and its Cheddar backend are compiled into `submission/orion`, with
Cheddar itself linked statically into the CUDA extension, so the suite's
requirement that a called library ship with the submission is met by the
tree and nothing has to be fetched or built. The implementation sources
are not included: the compiled objects are the deliverable.

**It requires a Blackwell GPU.** The CUDA extension is compiled for
`sm_120`. A machine with an older card cannot run it, which is a
deliberate restriction: the measurement machines behind the reported
numbers are Blackwell, as is the hardware the nearest competing
submission reports.

## Licensing

Because the submission ships built, the notices for the code compiled
into those binaries travel with them rather than with sources a reader
can inspect: `LICENSE-3RD-PARTIES.md`, beside this file, lists what
is in each object and under which terms, with the full texts under
`licenses/`. `LICENSE.md` beside it carries the terms for this submission's
own code: the MIT license, with portions copyright New York University and
LG Electronics USA, Inc. Cheddar and Orion are both MIT as well, which is
what permits their redistribution in binary form.

## Contact

**New York University (NYU) — Brooklyn Application, Architecture, and
Hardware Lab (BAAHL)**

- Akshath Mahajan, MSc — akshathmahajan@nyu.edu
- Prof. Brandon Reagen — bjr5@nyu.edu

**LG Electronics USA, Inc. — Emerging Tech Lab (ETL)**

- Geovandro Pereira, PhD
- Gabrielle de Micheli, PhD
- Henrique Seiti Ogawa
- Bahattin Yildiz, PhD

## Appendix

This section records detail that a reviewer may need but that is not
required in order to run the submission.

### Stage command interface

All four take the same command line as the Python stages they replace,
because the harness invokes either without knowing which it got
(`harness/utils.py:145`):

```
<stage> <size 0-3> [--count_only] [--seed N]
```

The size is the first argument that is not a flag. `--count_only`
selects Count and its absence selects Fetch. `--seed` is accepted and
ignored: the client's randomness is AES-CTR seeded from the operating
system, never from the harness's value, so a run cannot be made
reproducible by passing one. Any other `--flag` is skipped rather than
rejected, so that a future harness can pass arguments these stages do
not know about. A bad size prints
`usage: <stage> <size 0-3> [--count_only] [--seed N]` and exits.

Two properties are stated explicitly, as these binaries cannot be
inspected:

**No path comes from an argument or from the environment.** Each stage
derives the repository root from `/proc/self/exe`, walking up out of
`build/` and `submission/`, and composes every path it touches from
there — `io/<tier>` and `datasets/<tier>` inside the tree it was
shipped in. There is no argument that redirects a read or a write.

**`FBS_TRACE` is the only environment variable any of them reads.** Set
to anything, it adds per-phase timing lines to the output; it changes no
computation. Nothing else in the environment is consulted.

### Client stages

The client stages read the secret from the files key generation wrote,
and the public parameters from `.client/client.ctx` (written with the
keys). Their randomness is AES-128 in counter mode (ChaCha20 without
AES-NI), keyed from `getrandom` once per process. Key generation and database encryption stay
Python stages but do their arithmetic in `submission/build/libfbsnative.so`,
built by the same script; server setup is the Python shard pool.

### Module layout

Orion and its Cheddar backend live under `submission/orion`, so there is
nothing to place on `PYTHONPATH`. Each stage puts `submission/` on
`sys.path` itself, which resolves both `fbs` and `orion`. The subset is
deliberately small and imports no torch: `orion/__init__.py` resolves
its public API lazily, so the backend loads without the framework
behind it.

### Environment variables

| Variable | Effect |
|---|---|
| `FBS_SHARDS` | GPUs to shard across. Default: the **fewest** that keep the slice resident — the slice law against 85% of the card, capped at `ceil(K / 65536)` since sharding granularity is the block-group. A database that fits on one card is measured on one card; Large gets every card because no smaller count fits. A forced count still gets the residency decisions, at that count. |
| `FBS_COMPARATOR` | `gap5` / `gap6` / `gap8`, overriding the per-size default. |
| `FBS_N1` | BSGS baby step, overriding the per-size default. |
| `FBS_FETCH_CIRCUIT` | `direct` / `compact`, overriding the per-size default (direct through Medium, compact at Large). |
| `FBS_BOOT_FETCH` | `0` runs the compact circuit on the `shallow_40_L16` chain instead of bootstrapping — the conservative-parameters configuration, 886 bits of `log(QP)`. Default: bootstrap where the slice would not otherwise fit, which is Large alone. |
| `FBS_SEEDED_DB` | `0` expands the database's `a` on import instead of keeping the seed on the device. Default: the seed is kept only where it is required for residency. Every measurement of that change is against `FBS_SEEDED_DB=0`. |
| `FBS_QUERY_RING` | `subring` (default) / `seeded` / `full`. Forced to `full` under a bootstrap, because the subring switching key and the bootstrap both want rotation 24576. |
| `FBS_ENCRYPT_WORKERS` | client encryption processes. Default `min(nproc, 32)` — measured plateau. |
| `FBS_WARMUP` | warm-up queries each shard runs before accepting work. Default 3; the first three are measurably slower than the rest. |
| `FBS_NATIVE_CLIENT` | `0` falls back to the numpy client instead of the compiled one. Slower; it is the reference the compiled path is tested against. |
| `FBS_TRACE` | per-phase timings from the compute stage and the shards. |
| `FBS_PHASES` | per-phase GPU timings inside the query circuit. Synchronizes per phase, so it inflates `Compute`. |
