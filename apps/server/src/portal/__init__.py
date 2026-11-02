"""Independent CamS portal. Mount router under /api; initialize after env loading."""

from .router import router
from .service import initialize

__all__ = ["initialize", "router"]
