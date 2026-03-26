# One-shot warmup: PyReason and Numba JIT on first import; this module moves that
# cost off the first real move.
from __future__ import annotations

import chess

from .pyreason_move_selection import bootstrapPyreason, runPyreasonForMoves

# Dummy intervals so each predicate has a fact shape identical to a real turn.
_dummyInterval = (0.45, 0.55)
_predicateNamesUsedInRanking = (
    "Capture",
    "Check",
    "Center",
    "Safe",
    "TradeUp",
    "PinOk",
    "KingPress",
    "Cramp",
    "Develop",
    "UnderFire",
)


# Runs a tiny fake turn so PyReason and Numba compile once; real games feel less
# laggy after.
def runPrewarm(showProgressMessages: bool = True) -> None:
    if showProgressMessages:
        print("Importing PyReason (first launch may take ~30–90s on some machines)...")
    import pyreason as pyreasonModule  # noqa: PLC0415

    bootstrapPyreason(pyreasonModule)
    board = chess.Board()
    legalMoveList = list(board.legal_moves)
    openingMoveSample = legalMoveList[: min(len(legalMoveList), 12)]
    uciToPredicateFacts = {
        move.uci(): {predicateName: _dummyInterval for predicateName in _predicateNamesUsedInRanking}
        for move in openingMoveSample
    }
    if showProgressMessages:
        print("Running a sample reason() call to warm Numba helpers...")
    runPyreasonForMoves(pyreasonModule, openingMoveSample, uciToPredicateFacts)
    if showProgressMessages:
        print("Done — subsequent game launches should feel snappier.")


# If you run this file as main, just call runPrewarm with prints on.
def main() -> None:
    runPrewarm(showProgressMessages=True)


if __name__ == "__main__":
    main()
