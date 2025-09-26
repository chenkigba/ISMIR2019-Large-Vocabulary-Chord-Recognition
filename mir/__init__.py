try:
    # Try relative import when used as part of the package
    from .common import WORKING_PATH, PACKAGE_PATH
    from .data_file import TextureBuilder, DataEntry, DataPool
except ImportError:
    # Fall back to absolute import for standalone usage
    from mir.common import WORKING_PATH, PACKAGE_PATH
    from mir.data_file import TextureBuilder, DataEntry, DataPool


__all__ = [
    "TextureBuilder",
    "DataEntry",
    "WORKING_PATH",
    "PACKAGE_PATH",
    "DataPool",
    "io",
]
