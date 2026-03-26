# Evolutionary tuning of linear feature weights via mirror self-play and a scalar fitness
# score (checkmate outcome plus a small material bonus from White's perspective).
from __future__ import annotations

import argparse
import random
from typing import Dict, Tuple

import chess

from .chess_ai import PyReasonChessAI
from .weights import Weights, loadWeightsByDifficulty, saveWeightsByDifficulty

trainingMaterialValueByPieceType = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def whitePerspectiveMaterialScore(board: chess.Board) -> float:
    runningSum = 0.0
    for squareIndex in chess.SQUARES:
        piece = board.piece_at(squareIndex)
        if piece is None:
            continue
        value = trainingMaterialValueByPieceType.get(piece.piece_type, 0)
        runningSum += value if piece.color == chess.WHITE else -value
    return runningSum


def averageFitnessForWeights(weightVector: Weights, evaluationGameCount: int, randomSeed: int) -> float:
    # Fitness is mean over games of: terminal outcome (-1/0/+1 from White's eyes) plus
    # 0.02 times the signed material sum. Both players share the same weight vector.
    randomNumberGenerator = random.Random(randomSeed)
    cumulativeFitness = 0.0
    for _completedEvaluationGame in range(evaluationGameCount):
        board = chess.Board()
        whiteAgent = PyReasonChessAI(difficulty="medium")
        blackAgent = PyReasonChessAI(difficulty="medium")
        whiteAgent.weights = weightVector
        blackAgent.weights = weightVector

        maximumPliesPerEvaluationGame = 80
        for _halfmoveIndex in range(maximumPliesPerEvaluationGame):
            if board.is_game_over():
                break
            if board.turn == chess.WHITE:
                move, _notes = whiteAgent.chooseMove(board)
            else:
                move, _notes = blackAgent.chooseMove(board)
            board.push(move)

        outcomeScore = 0.0
        if board.is_checkmate():
            outcomeScore = 1.0 if board.turn == chess.BLACK else -1.0
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            outcomeScore = 0.0

        cumulativeFitness += outcomeScore + 0.02 * whitePerspectiveMaterialScore(board)
    return cumulativeFitness / float(max(1, evaluationGameCount))


def mutateWeightVector(sourceWeights: Weights, randomNumberGenerator: random.Random, mutationSigma: float) -> Weights:
    def addGaussianNoise(value: float) -> float:
        return value + randomNumberGenerator.gauss(0.0, mutationSigma)

    return Weights(
        bias=addGaussianNoise(sourceWeights.bias),
        captureValue=addGaussianNoise(sourceWeights.captureValue),
        givesCheck=addGaussianNoise(sourceWeights.givesCheck),
        isCenter=addGaussianNoise(sourceWeights.isCenter),
        hangsMovedPiece=addGaussianNoise(sourceWeights.hangsMovedPiece),
        winsMaterialAfterExchange=addGaussianNoise(sourceWeights.winsMaterialAfterExchange),
        pinnedMovedPiece=addGaussianNoise(sourceWeights.pinnedMovedPiece),
        kingPressure=addGaussianNoise(sourceWeights.kingPressure),
        crampOpponent=addGaussianNoise(sourceWeights.crampOpponent),
        developsOffBackRank=addGaussianNoise(sourceWeights.developsOffBackRank),
        underFireQuality=addGaussianNoise(sourceWeights.underFireQuality),
    )


def trainDifficultyWeights(
    difficultyKey: str,
    iterationCount: int,
    gamesPerEvaluation: int,
    mutationSigma: float,
    randomSeed: int,
) -> Tuple[Weights, float]:
    randomNumberGenerator = random.Random(randomSeed)
    existingByDifficulty = loadWeightsByDifficulty()
    bestWeights = existingByDifficulty.get(difficultyKey) or existingByDifficulty.get("medium") or Weights(
        0, 1.6, 1.0, 0.5, -2.0, 0.8, -1.1, 0.85, 0.55, 0.45, 1.0
    )
    bestFitness = averageFitnessForWeights(bestWeights, evaluationGameCount=gamesPerEvaluation, randomSeed=randomSeed + 11)

    for iterationIndex in range(iterationCount):
        candidateWeights = mutateWeightVector(bestWeights, randomNumberGenerator, mutationSigma=mutationSigma)
        candidateFitness = averageFitnessForWeights(
            candidateWeights, evaluationGameCount=gamesPerEvaluation, randomSeed=randomSeed + 101 + iterationIndex
        )
        if candidateFitness > bestFitness:
            bestWeights, bestFitness = candidateWeights, candidateFitness
            print(f"[{difficultyKey}] improved iter={iterationIndex} score={bestFitness:.3f} weights={bestWeights}")
        elif iterationIndex % 10 == 0:
            print(f"[{difficultyKey}] iter={iterationIndex} bestScore={bestFitness:.3f}")
    return bestWeights, bestFitness


def main() -> None:
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("--difficulty", choices=("easy", "medium", "hard", "achilles"), default="medium")
    argumentParser.add_argument("--trainingIterations", type=int, default=80)
    argumentParser.add_argument(
        "--evaluationGames",
        type=int,
        default=6,
        help="Games per fitness evaluation.",
    )
    argumentParser.add_argument("--mutationSigma", type=float, default=0.15, help="Mutation size for weight search.")
    argumentParser.add_argument("--randomSeed", type=int, default=42)
    commandLineArguments = argumentParser.parse_args()

    trainedWeights, finalFitness = trainDifficultyWeights(
        commandLineArguments.difficulty,
        commandLineArguments.trainingIterations,
        commandLineArguments.evaluationGames,
        commandLineArguments.mutationSigma,
        commandLineArguments.randomSeed,
    )

    weightsByDifficulty: Dict[str, Weights] = loadWeightsByDifficulty()
    weightsByDifficulty[commandLineArguments.difficulty] = trainedWeights
    saveWeightsByDifficulty(weightsByDifficulty)
    print(
        f"Saved trained weights for {commandLineArguments.difficulty} (score={finalFitness:.3f}) to src/trainedWeights.json"
    )


if __name__ == "__main__":
    main()
