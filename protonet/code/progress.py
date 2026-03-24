from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, TypeVar

from tqdm.auto import tqdm


T = TypeVar("T")


def track(iterable: Iterable[T], *, total: int | None = None, desc: str = "", enabled: bool = True) -> Iterator[T]:
    if not enabled:
        for item in iterable:
            yield item
        return
    yield from tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


@contextmanager
def task_bar(*, total: int, desc: str, enabled: bool = True):
    if not enabled:
        class _Dummy:
            def update(self, _: int = 1) -> None:
                return None

            def set_postfix(self, **_: object) -> None:
                return None

        yield _Dummy()
        return

    bar = tqdm(total=total, desc=desc, dynamic_ncols=True)
    try:
        yield bar
    finally:
        bar.close()


def announce(message: str) -> None:
    tqdm.write(message)
