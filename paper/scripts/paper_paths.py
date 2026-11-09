"""Paths shared by the standalone paper reproduction scripts."""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
REPO_DIR = PAPER_DIR.parent
DATASET_DIR = REPO_DIR / "datasets"
BLACKBOX_DIR = PAPER_DIR / "blackboxes"
