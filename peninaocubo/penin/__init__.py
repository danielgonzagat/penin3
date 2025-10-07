"""PENIN multi-API orchestration package."""

from __future__ import annotations

from importlib import metadata

from .config import settings  # noqa: F401
from .router import (  # noqa: F401
    MultiLLMRouterComplete as MultiLLMRouter,
    create_router_with_defaults,
)

try:  # pragma: no cover - resolved at runtime when package is installed
    __version__ = metadata.version("peninaocubo")
except metadata.PackageNotFoundError:  # pragma: no cover - local source tree
    __version__ = "0.9.0"  # IA AO CUBO Transformation - 85% Complete

__all__ = ["MultiLLMRouter", "create_router_with_defaults", "settings", "__version__"]
