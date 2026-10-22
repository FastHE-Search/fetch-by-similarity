#!/usr/bin/env python3
"""Encrypt the database, one file per shard.

Reads the harness's own db.bin rather than generating anything: this is
the first stage where the submission and the benchmark harness diverge,
because the benchmark draws a synthetic database from a seed and the
submission is handed one.

The slice is read with memmap, so a shard's rows cost their own bytes
and not the whole file -- at Large that file is 41 GB.

One ciphertext is resident at a time per worker: encode, encrypt,
export, free. The client's footprint is its key set plus that, whatever
the database size.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "libs"))
from fbs._common import (instance, open_client, read_db, read_payloads,
                     report_steps)


def main():
    inst = instance()
    inst.ensure_dirs()

    t0 = time.time()
    client = open_client(inst, adopt=True)
    client.open_encryptor(workers=inst.workers)
    total = 0
    try:
        for g, (lo, hi) in enumerate(inst.shard_ranges()):
            if lo == hi:
                raise SystemExit(
                    f"shard {g} owns no records: {inst.db_size:,} does not "
                    f"spread over {inst.shards} shards")
            rows = read_db(inst, lo, hi, raw=client.native is not None)
            path = os.path.join(inst.db_dir(), f"shard{g}.bin")
            t_s = time.time()
            # No fsync here. This stage did briefly force one, on the
            # theory that 18 GB of dirty page cache was flushing
            # underneath the query loop. It cost 152 s of Total (DB
            # encryption 50 s -> 203 s) and left Compute unchanged,
            # 0.143 s against 0.145 s -- which was already the evidence
            # that writeback pressure was not what made the query
            # stage's writes erratic. It was ext4 auto_da_alloc; see
            # write_ct in src/server/daemon.py.
            # Unlinked first rather than truncated by "wb": truncating a
            # file that still holds a previous run's 4.6 GB arms ext4's
            # auto_da_alloc, and close() then flushed all of it to the
            # device -- 40 s a shard, measured. The harness starts from an
            # empty io/, so only a re-run outside it paid that.
            for old in (path, os.path.join(inst.db_dir(), f"payload{g}.bin")):
                if os.path.exists(old):
                    os.unlink(old)
            with open(path, "wb") as fh:
                # CHCA only where the shard will keep the seed;
                # CHCS is ~2.4x cheaper to expand (seeded.py).
                n = client.stream_slice(rows, fh,
                                        addressable=inst.seeded_db)
            total += n
            del rows
            if not inst.count_only:
                # Fetch: each record's payload, laid out slot-for-slot
                # with the scores this shard will compute. Part of the
                # encrypted database, and counted in its bandwidth.
                ppath = os.path.join(inst.db_dir(), f"payload{g}.bin")
                with open(ppath, "wb") as fh:
                    total += client.stream_payloads(
                        read_payloads(inst, lo, hi), fh,
                        workers=inst.workers)
            print(f"  shard {g}: records [{lo:,}, {hi:,}) -> {n} "
                  f"ciphertexts in {time.time() - t_s:.1f}s", flush=True)
    finally:
        client.close_encryptor()

    elapsed = time.time() - t0
    how = (f"compiled, {os.cpu_count()} threads" if client.native
           else f"numpy, {inst.workers} workers")
    print(f"  {total} database ciphertexts in {elapsed:.1f}s "
          f"(one secret key, {how}, no GPU)")
    # "Client ...": see client_key_generation.py
    report_steps(inst, {"Client DB encryption": elapsed})
    return 0


if __name__ == "__main__":
    sys.exit(main())
