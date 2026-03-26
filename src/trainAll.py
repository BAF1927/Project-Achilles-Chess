# Runs one shared stream of self-play games while three Q-tables absorb different
# prefixes: easy stops after the first quarter of games, medium after half, hard keeps
# learning throughout.
from __future__ import annotations

import argparse
import random

from .chess_ai import PyReasonChessAI
from .qtable import QTable, loadQTable, saveQTable, updateQ
from .trainQ import playOneGame, terminalReward


def main() -> None:
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("--gameCount", type=int, default=200, help="Total shared self-play games")
    argumentParser.add_argument(
        "--playTime",
        type=int,
        default=120,
        help="Max half-moves per game before stopping (game length cap).",
    )
    argumentParser.add_argument("--randomSeed", type=int, default=42)
    argumentParser.add_argument(
        "--checkpointEvery",
        type=int,
        default=500,
        help="Save JSON checkpoints every N games (0 = end / interrupt only)",
    )
    argumentParser.add_argument(
        "--progressEvery",
        type=int,
        default=1,
        help="Print progress every N finished games (0 = silent)",
    )
    commandLineArguments = argumentParser.parse_args()

    totalGames = max(1, int(commandLineArguments.gameCount))
    easyStopAfterGame = max(1, int(totalGames * 0.25))
    mediumStopAfterGame = max(1, int(totalGames * 0.50))

    print(
        f"Shared training schedule: total={totalGames} easyStopsAt={easyStopAfterGame} mediumStopsAt={mediumStopAfterGame}"
    )

    memoryEasy: QTable = loadQTable("easy")
    memoryMedium: QTable = loadQTable("medium")
    memoryHard: QTable = loadQTable("hard")

    policyAgent = PyReasonChessAI(difficulty="hard")
    policyAgent.qtable = memoryHard  # type: ignore[attr-defined]

    random.seed(int(commandLineArguments.randomSeed))
    checkpointInterval = max(0, int(commandLineArguments.checkpointEvery))
    progressInterval = max(0, int(commandLineArguments.progressEvery))

    def writeAllTables(note: str) -> None:
        saveQTable(memoryEasy, difficulty="easy")
        saveQTable(memoryMedium, difficulty="medium")
        saveQTable(memoryHard, difficulty="hard")
        print(
            f"{note} | easy={len(memoryEasy.entries)} medium={len(memoryMedium.entries)} hard={len(memoryHard.entries)}",
            flush=True,
        )

    try:
        for gameIndex in range(totalGames):
            finalBoard, transitions = playOneGame(
                policyAgent,
                maximumPliesPerGame=commandLineArguments.playTime,
            )
            decayingTerminalReward = terminalReward(finalBoard)

            shouldUpdateEasy = (gameIndex + 1) <= easyStopAfterGame
            shouldUpdateMedium = (gameIndex + 1) <= mediumStopAfterGame
            shouldUpdateHard = True

            for positionKey, moveUci, nextPositionKey, shapingReward in reversed(transitions):
                combinedReward = shapingReward + decayingTerminalReward
                if shouldUpdateEasy:
                    updateQ(memoryEasy, positionKey, moveUci, reward=combinedReward, nextFen=nextPositionKey)
                if shouldUpdateMedium:
                    updateQ(memoryMedium, positionKey, moveUci, reward=combinedReward, nextFen=nextPositionKey)
                if shouldUpdateHard:
                    updateQ(memoryHard, positionKey, moveUci, reward=combinedReward, nextFen=nextPositionKey)
                decayingTerminalReward *= 0.98

            if progressInterval and (gameIndex + 1) % progressInterval == 0:
                print(f"Game {gameIndex + 1} finished", flush=True)
            if checkpointInterval and (gameIndex + 1) % checkpointInterval == 0:
                writeAllTables(f"Checkpoint game {gameIndex + 1}/{totalGames}")
    except KeyboardInterrupt:
        print("Interrupted — saving partial tables…")
    finally:
        writeAllTables("Saved qtables")
    print("Done (or stopped). Runtime loads these files and does not learn during play.", flush=True)


if __name__ == "__main__":
    main()
