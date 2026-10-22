"""Watchdog for the shard pool.

The pool used to re-execute _pool.py for this. That module is compiled,
and an extension module cannot be run as a script, so the entry point is
this file and the body lives in _pool.watchdog_main.

Kept as source because the harness's teardown never reaches a detached
daemon, so something has to outlive the run and stop the pool.
"""
import os
import sys

# The compiled objects are in libs/, beside this file. Python puts this
# script's own directory on sys.path, so libs/ has to be added by hand.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "libs"))

from fbs._pool import watchdog_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(watchdog_main(sys.argv))
