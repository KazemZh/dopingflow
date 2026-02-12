# src/dopingflow/logging.py
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path


def _suppress_external_noise() -> None:
    """
    Reduce noise from TensorFlow, DGL, etc.
    Must be called BEFORE importing heavy ML libs.
    """

    # TensorFlow suppression
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

    # DGL backend
    os.environ.setdefault("DGLBACKEND", "pytorch")

    # Silence noisy loggers
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("dgl").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def setup_logging(root: Path, *, verbose: bool = False) -> Path:
    """
    Configure logging for dopingflow.

    - Console output
    - Per-run log file
    - Noise suppression
    """

    _suppress_external_noise()

    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logfile = log_dir / f"dopingflow_{timestamp}.log"

    level = logging.DEBUG if verbose else logging.INFO

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    logging.getLogger(__name__).info("Logging initialized")
    logging.getLogger(__name__).info("Log file: %s", logfile)

    return logfile
