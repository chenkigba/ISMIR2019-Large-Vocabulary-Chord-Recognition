"""
LVCR_ismir2019 - Large-Vocabulary Chord Recognition
"""
from pathlib import Path

from .chord_recognition import chord_recognition, chord_recognition_from_memory

__version__ = "1.0.0"
__all__ = ["chord_recognition", "chord_recognition_from_memory", "PACKAGE_ROOT", "get_resource_path"]

PACKAGE_ROOT = Path(__file__).resolve().parent


def get_resource_path(*relative_parts: str) -> Path:
    """Return an absolute path inside the installed package."""
    return PACKAGE_ROOT.joinpath(*relative_parts)
