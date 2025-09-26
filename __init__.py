"""
ISMIR2019 Large Vocabulary Chord Recognition

This package provides chord recognition functionality using the method described in:
"Large-Vocabulary Chord Transcription via Chord Structure Decomposition" (ISMIR 2019)
"""

__version__ = "1.0.0"
__author__ = "Junyan Jiang"

# Import main modules
from . import mir
from . import extractors
from . import io_new

# Import main functions
try:
    from .chord_recognition import chord_recognition_main
except ImportError:
    # Handle case where dependencies might not be available
    chord_recognition_main = None

# Also make it available at module level for direct import
try:
    from chord_recognition import chord_recognition_main as _chord_recognition_main

    if chord_recognition_main is None:
        chord_recognition_main = _chord_recognition_main
except ImportError:
    pass

__all__ = [
    "mir",
    "extractors",
    "io_new",
    "chord_recognition_main",
]
