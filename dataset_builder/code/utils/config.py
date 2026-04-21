from __future__ import annotations

try:
    from .contracts import BuilderConfig
except ImportError:  # pragma: no cover
    from contracts import BuilderConfig
