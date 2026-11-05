#!/usr/bin/env python3
"""Bring the shard pool up, once, before the query loop.

The harness runs each stage as a separate process, so a server that
loaded its evaluation keys and encrypted database per query would pay
that load inside the stage reported as `Compute`. This stage pays it
once: the daemons stay parked on a request FIFO, and each
`server_encrypted_compute` sends them one query.

A detached watchdog stops them when the harness exits, which keeps the
lifecycle inside the submission -- the harness has no teardown step and
the rules forbid modifying it.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "libs"))
from fbs import _pool
from fbs._common import instance, report_steps


def main():
    inst = instance()
    inst.ensure_dirs()
    _pool.stop(inst)                       # any pool left by an earlier run

    t0 = time.time()
    shard = _pool.start(inst)               # the slowest shard's timers
    elapsed = time.time() - t0

    # The sizing that chose this shard count, printed so a card the
    # slice law mispredicts is visible in the run rather than silent.
    from fbs._common import _gpu_memory_mib
    card = _gpu_memory_mib()
    fit = (f", {inst.peak_mib / card:.0%} of a {card:,} MiB card"
           if card else "")
    print(f"  {inst.shards} shard(s) loaded and ready in {elapsed:.1f}s "
          f"({inst.ct_per_shard:,} ct/shard, predicted peak "
          f"{inst.peak_mib:,.0f} MiB{fit})")
    # "Encrypted dataset preprocessing" is the harness's own name for this
    # stage, and the leaderboard fills its server Enc Prep column by that
    # name (see README.md): the pool coming up -- keys, database
    # regeneration and import, warm-up -- is exactly that, measured here
    # without this interpreter's own start.
    #
    # Below it, where that time went, so loading can be told apart from
    # computing. The parts are disjoint and add up to the total: the
    # slowest shard's own timers, with whatever they cannot see --
    # spawning the process, its interpreter starting, the READY reaching
    # us -- booked to its start.
    seen = sum(shard.values())
    steps = {
        "Encrypted dataset preprocessing": elapsed,
        "Setup: shard process start":
            shard["start"] + max(0.0, elapsed - seen),
        "Setup: load evaluation keys": shard["keys"],
        "Setup: load encrypted database": shard["db"],
        "Setup: engine construction": shard["engine"],
        "Setup: warm-up computation": shard["warm"],
    }
    for name, t in list(steps.items())[1:]:
        print(f"    {name[7:]:<24} {t:8.3f}s")
    report_steps(inst, steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
