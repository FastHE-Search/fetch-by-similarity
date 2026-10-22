"""Launcher for one shard.

The pool used to start a shard with `python -m fbs.server.daemon`.
Compiled modules cannot be executed that way -- runpy needs a source or
bytecode loader -- so the entry point is this file instead. It stays as
source because it is three lines and holds nothing: the circuit is in
the compiled module it calls.

The pool's watchdog identifies a stray shard by this file's name in
/proc/<pid>/cmdline, so do not rename it without changing _pool.py.
"""
import os
import sys

# The compiled objects are in libs/, beside this file. Python puts this
# script's own directory on sys.path, so libs/ has to be added by hand.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "libs"))

from fbs.server.daemon import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
