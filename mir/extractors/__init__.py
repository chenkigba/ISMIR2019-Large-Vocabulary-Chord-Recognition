try:
    # Try relative import when used as part of the package
    from .extractor_base import ExtractorBase
except ImportError:
    # Fall back to absolute import for standalone usage
    from mir.extractors.extractor_base import ExtractorBase

__all__ = ["ExtractorBase"]
