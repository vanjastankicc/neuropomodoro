# -*- coding: utf-8 -*-
"""
muse_stream.py - Starts the Muse 2 LSL stream from within Python
-----------------------------------------------------------------

Launches `muselsl stream` as a background process, waits until the EEG stream
is available on the network and closes the process when the script exits.
If an EEG stream is already running (e.g. started manually in a terminal),
no new process is launched.
"""

import atexit
import shutil
import subprocess
import sys
import time

from pylsl import resolve_byprop

STREAM_TIMEOUT = 30   # seconds to wait for the headband to connect

_process = None


def _stop_stream():
    """Closes the muselsl process if this module started it."""
    if _process is not None and _process.poll() is None:
        print("Closing Muse stream...")
        _process.terminate()
        try:
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _process.kill()


def start_muse_stream():
    """Starts `muselsl stream` and returns the EEG stream info once available."""
    global _process

    streams = resolve_byprop("type", "EEG", timeout=2)
    if streams:
        print("EEG stream already running.")
        return streams

    # Use the muselsl command if it is on PATH, otherwise run it as a module
    if shutil.which("muselsl"):
        command = ["muselsl", "stream"]
    else:
        command = [sys.executable, "-m", "muselsl", "stream"]

    print("Starting Muse stream...")
    _process = subprocess.Popen(command)
    atexit.register(_stop_stream)

    start = time.time()
    while time.time() - start < STREAM_TIMEOUT:
        if _process.poll() is not None:
            raise RuntimeError(
                "muselsl stopped. Check that the Muse is on and Bluetooth is enabled."
            )
        streams = resolve_byprop("type", "EEG", timeout=2)
        if streams:
            return streams

    _stop_stream()
    raise RuntimeError(f"No EEG stream found within {STREAM_TIMEOUT} s.")
