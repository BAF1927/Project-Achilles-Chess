# Entry point: builds ChessApplication and runs the pygame loop until quit.
from __future__ import annotations

from .ui import ChessApplication


def main() -> None:
    ChessApplication().run()


if __name__ == "__main__":
    main()
