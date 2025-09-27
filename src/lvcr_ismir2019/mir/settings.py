"""Runtime settings for MIR utilities."""
import os

SONIC_VISUALIZER_PATH = os.getenv(
    'LVCR_SONIC_VISUALIZER_PATH', 'sonic-visualiser'
)
SONIC_ANNOTATOR_PATH = os.getenv(
    'LVCR_SONIC_ANNOTATOR_PATH', 'sonic-annotator'
)
DEFAULT_DATA_STORAGE_PATH = os.getenv(
    'LVCR_DEFAULT_DATA_STORAGE_PATH', os.path.join(os.getcwd(), 'cache_data')
)
