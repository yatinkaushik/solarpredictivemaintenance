"""
logger.py — Centralised logging for Solar AI
Replaces all print() calls with structured, levelled log output.
Logs go to both the console (INFO+) and a rotating file (DEBUG+).
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from config import LOG_PATH

def get_logger(name: str = "solar_ai") -> logging.Logger:
    """
    Returns a configured logger. Safe to call multiple times —
    handlers are only added once per logger name.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger   # already configured

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── Console handler (INFO and above) ────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── Rotating file handler (DEBUG and above, max 2 MB × 3 backups) ───────
    try:
        fh = RotatingFileHandler(
            LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not create file log handler: {e}")

    return logger