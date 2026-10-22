#!/usr/bin/env python3
"""Optional plaintext work on the query. We need none.

Normalization happens inside the query encoder, on the same vector it
is about to encrypt, so doing it here would only write a second copy of
a 128-to-512 element array to disk.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "libs"))
from fbs._common import instance


def main():
    inst = instance()
    q = os.path.join(inst.data_dir(), "query.bin")
    if not os.path.exists(q):
        raise SystemExit(f"{q} missing")
    expect = inst.dim * 4
    got = os.path.getsize(q)
    if got != expect:
        raise SystemExit(f"{q} is {got} bytes, expected {expect}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
