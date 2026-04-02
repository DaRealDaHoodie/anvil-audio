"""
stdio guard utilities.

MCP servers communicate with clients over stdio — the JSON-RPC message stream
lives on stdout, and anything else written there corrupts the protocol.  All
status/progress output must go to stderr instead.

``stdout_to_stderr()`` is a context manager that temporarily reassigns
``sys.stdout`` to ``sys.stderr`` for the duration of the block.  Use it to
wrap third-party library calls that print progress messages to stdout (model
loading printouts, weight conversion progress, HuggingFace Hub download bars,
etc.).

Usage::

    from anvil_audio.utils.stdio_guard import stdout_to_stderr

    with stdout_to_stderr():
        heavy_library.load(...)   # any print() inside goes to stderr
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Generator


@contextmanager
def stdout_to_stderr() -> Generator[None, None, None]:
    """Redirect ``sys.stdout`` to ``sys.stderr`` for the duration of the block.

    Thread safety: this swaps a module-level attribute; do **not** use in
    concurrent contexts where multiple threads share the same stdout.  In
    practice all Anvil pipeline loading is single-threaded.
    """
    _original = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = _original
