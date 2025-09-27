import os
from pathlib import Path

from .settings import *

PACKAGE_PATH = Path(__file__).resolve().parent
PACKAGE_ROOT = PACKAGE_PATH.parent
WORKING_PATH = os.environ.get("LVCR_WORKING_PATH", str(PACKAGE_ROOT))
DEFAULT_DATA_STORAGE_PATH = DEFAULT_DATA_STORAGE_PATH.replace(
    '$', PACKAGE_ROOT.name
)
DATA_PATH = PACKAGE_ROOT / 'data'
CACHE_PATH = PACKAGE_ROOT / 'cache_data'
