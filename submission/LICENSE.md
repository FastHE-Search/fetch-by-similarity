# License

MIT License

Portions copyright (c) 2026 Brooklyn Application, Architecture, and Hardware Lab, New York University
Portions copyright (c) 2026 LG Electronics USA, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## What this covers

Everything in `submission/` that is ours:

| Path | Description |
|---|---|
| `libs/fbs.cpython-310-*.so` | the circuit, compiled |
| `libs/orion.cpython-310-*.so` | the backend, compiled |
| `libs/cheddar_native32.cpython-310-*.so` | the CUDA extension, with Cheddar linked in |
| `libs/libfbsnative.so` | the client's and shard's hot loops |
| `build/*` | the four stage programs the harness invokes |
| `*.py` | the stage shims, as source |
| `data/*` | the prime chains and the bootstrap rotation list |
| `README.md` | the account of what the implementation is and assumes |

## What it does not cover

**Third-party code compiled into those binaries** is covered by its own
terms, reproduced in full under `licenses/` and listed in
[`LICENSE-3RD-PARTIES.md`](LICENSE-3RD-PARTIES.md). Cheddar and Orion
are both MIT, which is what permits redistributing them in binary form;
the rest are MIT, BSD-3-Clause, Apache-2.0 and the PSF license. Nothing
in this file overrides any of them.

**The benchmark harness** — everything outside `submission/` — is the
[fetch-by-similarity][u] project's, under the Apache License 2.0 in
`LICENSE.md` at the repository root. `harness/` is byte-identical to
upstream and unmodified.

[u]: https://github.com/fhe-benchmarking/fetch-by-similarity
