"""Locations for historical research scripts; override before starting Python."""
import os
from pathlib import Path


def location(variable, default):
    return Path(os.environ.get(variable) or default).expanduser().resolve()


ROOT = location('COTIGER_ROOT', Path(__file__).resolve().parents[1])
BASE = location('COTIGER_BEAUTY_BASE', ROOT/'out/cloud_reproduction/tiger_beauty_fast_20261012_194804')
STUDY = location('COTIGER_STUDY_ROOT', ROOT/'out/tiger_prefix_ft/beauty_inference_optimization_20261016')
RAW_ROOT = location('COTIGER_RAW_ROOT', ROOT/'out/cloud_reproduction/beauty_tiger_raw_20261013/full')
PREFIX_DATA = location('COTIGER_PREFIX_DATA', ROOT/'out/tiger_prefix_ft/beauty_20261013/data')
