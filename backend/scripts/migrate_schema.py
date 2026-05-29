from __future__ import annotations

import logging

from core.bootstrap import bootstrap_database


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    bootstrap_database()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
