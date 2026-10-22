#!/usr/bin/env python3
"""Sample the secret, provision the rotation keys, export the evks.

This is the only stage that samples a secret key. It writes it to
io/<inst>/.client/, which is what lets the later client stages -- a
separate process each -- encrypt and decrypt with it. The file IS the
key rather than a seed it could be regenerated from: storing a seed
would keep something equally sensitive on disk while also making the
error sampling reproducible.

Only the EVALUATION keys go to io/<inst>/keys/, which the harness `du`s
and reports as key bandwidth. The secret is not sent anywhere and is not
counted as if it were.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "libs"))
from fbs._common import (instance, open_client, report_steps,
                         write_stage_context)


def main():
    inst = instance()
    inst.ensure_dirs()

    t0 = time.time()
    client = open_client(inst, adopt=False)
    client.persist_secret()
    shifts = client.provision_and_export_keys(os.path.join(inst.keys_dir(), "evk.bin"))
    # The compiled per-query stages read this rather than the chain.
    write_stage_context(inst, client)
    elapsed = time.time() - t0

    print(f"  {len(shifts)} rotation keys provisioned for "
          f"{inst.comparator} at dim {inst.dim}")
    # "Client ..." so the key cannot match a leaderboard column: the board
    # fills its server-timing columns from this file by step name.
    report_steps(inst, {"Client key generation": elapsed})
    return 0


if __name__ == "__main__":
    sys.exit(main())
