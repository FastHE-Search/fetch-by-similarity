#!/usr/bin/env python3
"""Optional plaintext work on the dataset. We need none.

The packing this submission uses -- normalize, diagonalize, pre-rotate
the BSGS giant steps -- is fused into the encryption stage, because it
produces exactly the rows that get encoded and keeping them would mean
writing the plaintext database out a second time.

So this stage only checks that the dataset is the shape the instance
says it is, which turns a mismatch into a message here rather than a
reshape error three stages later.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "libs"))
from fbs._common import PAYLOAD_DIM, instance


def main():
    inst = instance()
    inst.ensure_dirs()
    db = os.path.join(inst.data_dir(), "db.bin")
    if not os.path.exists(db):
        raise SystemExit(f"{db} missing; run the harness, not this script")
    expect = inst.db_size * inst.dim * 4          # float32
    got = os.path.getsize(db)
    if got != expect:
        raise SystemExit(
            f"{db} is {got:,} bytes, expected {expect:,} for "
            f"{inst.db_size:,} x {inst.dim} float32")
    print(f"  {inst.name}: {inst.db_size:,} records of dim {inst.dim}, "
          f"comparator {inst.comparator}, {inst.shards} shard(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
