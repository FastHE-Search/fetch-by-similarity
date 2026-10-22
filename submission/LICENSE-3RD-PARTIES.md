# Third-party notices

This submission is distributed as compiled objects. Code from the
projects below is compiled into them, so their notices travel with the
binaries rather than with sources a reader can inspect. This file is
that notice.

Our own code is covered by [`LICENSE.md`](LICENSE.md) beside this file.
The Apache-2.0 text in `LICENSE.md` at the repository root is the
benchmark harness's, not ours.

The list was produced by reading what is actually in the artifacts, not
by listing what the build mentions: symbol and string inspection of
`submission/libs/*.so` and of `libcheddar.a`, and `ldd` for what is
loaded rather than embedded. Two projects the build fetches are
therefore absent — googletest, which only builds Cheddar's unit tests,
and jsoncpp, which has no symbols in the archive we link.

## Compiled into the shipped objects

| Project | Version | License | Linked into |
|---|---|---|---|
| [Cheddar](https://github.com/scale-snu/cheddar-fhe) | `8df8b26` (2026-04-09), patched | MIT | `cheddar_native32…so`, linked statically |
| [Orion](https://github.com/baahl-nyu/orion) | our fork, `cheddar-orion` | MIT | `orion…so` |
| [pybind11](https://github.com/pybind/pybind11) | 3.1.0 | BSD-3-Clause | `cheddar_native32…so` |
| [RMM](https://github.com/rapidsai/rmm) | 22.12.00 (`be379e5f`) | Apache-2.0 | `cheddar_native32…so`, header-only |
| [Thrust](https://github.com/NVIDIA/thrust) and CUB | 1.17.2 | Apache-2.0 | `cheddar_native32…so`, header-only |
| [spdlog](https://github.com/gabime/spdlog) | 1.9.2 (`libspdlog-dev` 1:1.9.2+ds-0.2) | MIT | `cheddar_native32…so`, header-only; required by RMM |
| [fmt](https://github.com/fmtlib/fmt) | 8.1.1 (`libfmt-dev` 8.1.1+ds1-2) | MIT | `cheddar_native32…so`. spdlog here is the distribution build, which uses the system fmt rather than a bundled copy, and we compile it with `FMT_HEADER_ONLY` so the code is inlined into the object instead of adding a shared dependency |
| [Nuitka](https://github.com/Nuitka/Nuitka) | 4.2.2 | Apache-2.0 | `fbs…so` and `orion…so`; its runtime is part of every module it compiles |
| [CPython](https://github.com/python/cpython) | 3.10.12 headers | PSF-2.0 | all three extension modules |

Apache-2.0 requires that its notices be retained and that modifications
be stated. Cheddar is patched: seven patches, described in the
development repository. RMM, Thrust, CUB and Nuitka are used unmodified.

## Loaded at run time, not shipped

These are not redistributed with the submission. The environment
provides them, and `submission/Dockerfile` installs them from their
usual packages.

| Project | License | Usage |
|---|---|---|
| [GMP](https://gmplib.org/) 6.2.1 (`libgmp10` 2:6.2.1+dfsg-3ubuntu1) | LGPL-3.0-or-later / GPL-2.0-or-later, at your option | dynamically linked, `libgmp.so.10`. Linked shared rather than static because the distribution's static library is built without `-fPIC`, which also keeps the LGPL's relinking provision straightforwardly satisfied. Both licence texts are included, since the packaged notice only points at them |
| CUDA runtime 12.8 | NVIDIA CUDA EULA | dynamically linked, `libcudart.so.12`. Cheddar's RMM additionally `dlopen`s it by name |
| [NumPy](https://numpy.org/) 2.2.6 | BSD-3-Clause | imported by the Python stages; `requirements.txt` installs it, and no NumPy code is compiled into our objects |

The container image additionally derives from `nvidia/cuda`, whose
contents are governed by the NVIDIA Deep Learning Container License.

## Not third-party

The Keccak permutation and SHAKE-128 in `libfbsnative.so` and in the
compiled stage programs are our own implementation, written from FIPS
202 with the round table unrolled. No reference implementation was
copied, and there is no notice to carry.

---

The license texts are reproduced in full under `licenses/`, one file
per project, named for the project and its licence. Nuitka ships three:
the Apache-2.0 licence, its NOTICE, and the separate terms covering the
runtime parts that become part of every module it compiles, which is the
part that applies to us.

Each text was taken from the copy the build actually used — the
dependency tree Cheddar fetched, the installed distribution packages,
the Python wheels — rather than from upstream repositories, so the
notices match the versions in the binaries.

This file records what is in the artifacts and under which terms. It is
not legal advice, and the obligations should be confirmed by counsel
before publication.
