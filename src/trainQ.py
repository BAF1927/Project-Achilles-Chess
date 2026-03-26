# Self-play Q-table trainer: each move still comes from PyReasonChessAI, then Bellman
# backups walk backward over the recorded (state, action, reward) transitions when the
# game ends.
from __future__ import annotations

import argparse
import random
from typing import List, Tuple

import chess

from .chess_ai import PyReasonChessAI
from .qtable import QTable, loadQTable, normalizeFenKey, saveQTable, updateQ
from .training_shaping import applyAchillesShaping

materialValueByPieceTypeForTraining = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


# Signed material count: positive favours White, using training weights (king counts as zero).
def whitePerspectiveMaterialSum(board: chess.Board) -> float:
    runningTotal = 0.0
    for squareIndex in chess.SQUARES:
        piece = board.piece_at(squareIndex)
        if piece is None:
            continue
        pieceValue = materialValueByPieceTypeForTraining.get(piece.piece_type, 0)
        runningTotal += pieceValue if piece.color == chess.WHITE else -pieceValue
    return runningTotal


def terminalOutcomeRewardWhitePerspective(board: chess.Board) -> float:
    if board.is_checkmate():
        return 1.0 if board.turn == chess.BLACK else -1.0
    return 0.0


def playOneGame(
    artificialIntelligence: PyReasonChessAI,
    maximumPliesPerGame: int,
    trainingDepth: str = "standard",
) -> Tuple[chess.Board, List[Tuple[str, str, str, float]]]:
    # trainingDepth:
    #   'standard' - only 0.03 times the material delta each ply.
    #   'achilles' - after the game, applyAchillesShaping adds shuttle / under-fire /
    #   major-loss hindsight.
    board = chess.Board()
    transitionList: List[Tuple[str, str, str, float]] = []
    previousMaterial = whitePerspectiveMaterialSum(board)

    for _halfmoveIndex in range(maximumPliesPerGame):
        if board.is_game_over():
            break
        positionKey = normalizeFenKey(board.fen())
        chosenMove, _notes = artificialIntelligence.chooseMove(board)
        board.push(chosenMove)
        nextPositionKey = normalizeFenKey(board.fen())
        currentMaterial = whitePerspectiveMaterialSum(board)
        materialDelta = currentMaterial - previousMaterial
        previousMaterial = currentMaterial
        transitionList.append((positionKey, chosenMove.uci(), nextPositionKey, 0.03 * materialDelta))

    if trainingDepth == "achilles":
        transitionList = applyAchillesShaping(transitionList)
    return board, transitionList


def trainSingleDifficultyQTable(difficultyLevel: str, gameCount: int, maximumPliesPerGame: int, randomSeed: int) -> None:
    random.seed(randomSeed)
    memoryTable = loadQTable(difficultyLevel)
    artificialIntelligence = PyReasonChessAI(difficulty=difficultyLevel)
    artificialIntelligence.qtable = memoryTable  # type: ignore[attr-defined]

    for completedGames in range(gameCount):
        finalBoard, transitions = playOneGame(
            artificialIntelligence,
            maximumPliesPerGame=maximumPliesPerGame,
        )
        decayingTerminalReward = terminalOutcomeRewardWhitePerspective(finalBoard)
        for positionKey, moveUci, nextPositionKey, shapingReward in reversed(transitions):
            combinedReward = shapingReward + decayingTerminalReward
            updateQ(memoryTable, positionKey, moveUci, reward=combinedReward, nextFen=nextPositionKey)
            decayingTerminalReward *= 0.98
        if (completedGames + 1) % 5 == 0:
            print(f"qtrain games={completedGames + 1} tableStates={len(memoryTable.entries)}")

    saveQTable(memoryTable, difficulty=difficultyLevel)
    print(f"Saved Q-table to src/qtable_{difficultyLevel}.json with states={len(memoryTable.entries)}")


def main() -> None:
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("--difficulty", choices=("easy", "medium", "hard", "achilles"), default="medium")
    argumentParser.add_argument("--gameCount", type=int, default=50)
    argumentParser.add_argument(
        "--playTime",
        type=int,
        default=80,
        help="Max half-moves per game before stopping (game length cap).",
    )
    argumentParser.add_argument("--randomSeed", type=int, default=42)
    commandLineArguments = argumentParser.parse_args()
    trainSingleDifficultyQTable(
        difficultyLevel=commandLineArguments.difficulty,
        gameCount=commandLineArguments.gameCount,
        maximumPliesPerGame=commandLineArguments.playTime,
        randomSeed=commandLineArguments.randomSeed,
    )


terminalReward = terminalOutcomeRewardWhitePerspective

if __name__ == "__main__":
    main()
