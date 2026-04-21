from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

try:
    from .contracts import BuilderConfig
except ImportError:  # pragma: no cover
    # Support running this module directly from within dataset_builder/code.
    from contracts import BuilderConfig


PipelineCallable = Callable[[BuilderConfig], Any]


def run_pipeline_sync(cfg: BuilderConfig, *, pipeline: PipelineCallable | None = None) -> dict[str, Any]:
    """Run the dataset pipeline from synchronous experiment scripts.

    `build_dataset.run_pipeline` is async in the current implementation. Keeping
    the await boundary here prevents sweep/ablation callers from accidentally
    treating a coroutine as a report dictionary.
    """
    if pipeline is None:
        try:
            from .build_dataset import run_pipeline
        except ImportError:  # pragma: no cover
            from build_dataset import run_pipeline

        pipeline = run_pipeline

    result = pipeline(cfg)
    if inspect.isawaitable(result):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(result)
        if inspect.iscoroutine(result):
            result.close()
        raise RuntimeError("run_pipeline_sync cannot be called while an event loop is already running")
    if not isinstance(result, dict):
        raise TypeError(f"Pipeline returned {type(result).__name__}; expected dict report")
    return result
