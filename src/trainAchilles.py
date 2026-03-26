# Train the Achilles Q-table (src/qtable_achilles.json) with deeper reward shaping than
# easy/medium/hard: see training_shaping.applyAchillesShaping and playOneGame(...,
# trainingDepth="achilles") in trainQ.py.
from __future__ import annotations

import argparse
import random

from .chess_ai import PyReasonChessAI
from .qtable import QTable, loadQTable, saveQTable, updateQ
from .trainQ import playOneGame, terminalReward


def main() -> None:
    p = argparse.ArgumentParser(description="Train Achilles → src/qtable_achilles.json only.")
    p.add_argument("--gameCount", type=int, default=500)
    p.add_argument(
        "--playTime",
        type=int,
        default=120,
        help="Max half-moves per game before stopping (game length cap).",
    )
    p.add_argument("--randomSeed", type=int, default=42)
    p.add_argument("--checkpointEvery", type=int, default=500)
    p.add_argument("--progressEvery", type=int, default=1)
    args = p.parse_args()

    total = max(1, int(args.games))
    ck = max(0, int(args.checkpointEvery))
    pe = max(0, int(args.progressEvery))

    q: QTable = loadQTable("achilles")
    ai = PyReasonChessAI(difficulty="achilles")
    ai.qtable = q  # type: ignore[attr-defined]
    random.seed(int(args.randomSeed))

    def writeTable(note: str) -> None:
        saveQTable(q, difficulty="achilles")
        print(f"{note} | states={len(q.entries)}", flush=True)

    print(
        f"Achilles: gameCount={total} playTime={args.playTime} randomSeed={args.randomSeed} checkpoint={ck or 'end-only'}",
        flush=True,
    )
    try:
        for g in range(total):
            board, transitions = playOneGame(
                ai,
                maximumPliesPerGame=args.playTime,
                trainingDepth="achilles",
            )
            rTerm = terminalReward(board)
            for (fen, uci, nextFen, shape) in reversed(transitions):
                reward = shape + rTerm
                updateQ(q, fen, uci, reward=reward, nextFen=nextFen)
                rTerm *= 0.98
            if pe and (g + 1) % pe == 0:
                print(f"Game {g + 1} finished", flush=True)
            if ck and (g + 1) % ck == 0:
                writeTable(f"Checkpoint game {g + 1}/{total}")
    except KeyboardInterrupt:
        print("Ctrl+C — saving…", flush=True)
    finally:
        writeTable("Saved qtable_achilles.json")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
