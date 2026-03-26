# Runtime move filters plus extra reward shaping used only for Achilles Q-learning.
# All difficulties: chooseMove may remove shuffle-undo and same-piece shuttle patterns.
# Achilles training only: after each self-play game, transition rewards are adjusted for
# Black shuttles, weak handling of attacked majors, and hindsight when White trades into
# a Black major.
from __future__ import annotations

from typing import List, Tuple

import chess

from .features import computeMoveFeatures, pieceValuesByType

TransitionTuple = Tuple[str, str, str, float]


# Rewinds a board copy to the ply before move_stack[stackIndex], then asks python-chess
# if that move was a capture.
def stackMoveWasCaptureWhenPlayed(board: chess.Board, stackIndex: int) -> bool:
    totalPlies = len(board.move_stack)
    if stackIndex < 0 or stackIndex >= totalPlies:
        return False
    boardCopy = board.copy()
    for _undoStep in range(totalPlies - stackIndex):
        boardCopy.pop()
    stackedMove = board.move_stack[stackIndex]
    return boardCopy.is_capture(stackedMove)


# Detects a fourth non-capture leg where the same piece oscillates between two squares
# (A-B-A-B-B-A pattern), using the last three moves by the same color from the stack.
def isPieceShuttleMove(board: chess.Board, move: chess.Move) -> bool:
    moveHistory = board.move_stack
    if len(moveHistory) < 6:
        return False
    indexMostRecentOwnMove = len(moveHistory) - 2
    mostRecentOwnMove = moveHistory[indexMostRecentOwnMove]
    if board.is_capture(move) or stackMoveWasCaptureWhenPlayed(board, indexMostRecentOwnMove):
        return False
    if mostRecentOwnMove.to_square != move.from_square or move.to_square != mostRecentOwnMove.from_square:
        return False
    if indexMostRecentOwnMove - 2 < 0:
        return False
    secondPreviousOwnMove = moveHistory[indexMostRecentOwnMove - 2]
    if stackMoveWasCaptureWhenPlayed(board, indexMostRecentOwnMove - 2):
        return False
    if secondPreviousOwnMove.to_square != mostRecentOwnMove.from_square or secondPreviousOwnMove.from_square != mostRecentOwnMove.to_square:
        return False
    if indexMostRecentOwnMove - 4 < 0:
        return False
    thirdPreviousOwnMove = moveHistory[indexMostRecentOwnMove - 4]
    if stackMoveWasCaptureWhenPlayed(board, indexMostRecentOwnMove - 4):
        return False
    if (
        thirdPreviousOwnMove.to_square != secondPreviousOwnMove.from_square
        or thirdPreviousOwnMove.from_square != secondPreviousOwnMove.to_square
    ):
        return False
    return True


# Removes shuttle candidates; if that would empty the set, returns the original moves
# so the game never deadlocks.
def filterOutPieceShuttleMoves(board: chess.Board, legalMoves: list[chess.Move]) -> list[chess.Move]:
    keptMoves = [legalMove for legalMove in legalMoves if not isPieceShuttleMove(board, legalMove)]
    return keptMoves if keptMoves else legalMoves


def applyAchillesShaping(
    transitions: List[TransitionTuple],
    lookbackBlackPlies: int = 5,
) -> List[TransitionTuple]:
    # Adds shaping on top of the per-ply 0.03 * material delta collected during self-play.
    # Black shuttles, bad under-fire replies, and White's capture of a Black major get
    # extra penalties spread backward.
    replayBoard = chess.Board()
    rewardAdjustmentByIndex = [0.0] * len(transitions)

    for transitionIndex, (_positionKey, moveUci, _nextKey, _baseReward) in enumerate(transitions):
        moveObject = chess.Move.from_uci(moveUci)
        sideToMoveBeforePush = replayBoard.turn

        if sideToMoveBeforePush == chess.BLACK and isPieceShuttleMove(replayBoard, moveObject):
            rewardAdjustmentByIndex[transitionIndex] -= 0.09

        if sideToMoveBeforePush == chess.BLACK:
            underFireScore = computeMoveFeatures(replayBoard, moveObject).underFireQuality
            if underFireScore >= 0.88:
                rewardAdjustmentByIndex[transitionIndex] += 0.04
            elif underFireScore <= 0.24:
                rewardAdjustmentByIndex[transitionIndex] -= 0.05

        blackLostMajorPiece = False
        victimMaterialValue = 0.0
        if replayBoard.is_en_passant(moveObject) and sideToMoveBeforePush == chess.WHITE:
            pass
        else:
            victimPiece = replayBoard.piece_at(moveObject.to_square)
            if victimPiece is not None:
                victimMaterialValue = float(pieceValuesByType.get(victimPiece.piece_type, 0))
                blackLostMajorPiece = victimPiece.color == chess.BLACK and victimMaterialValue >= 5.0

        replayBoard.push(moveObject)

        if sideToMoveBeforePush == chess.WHITE and blackLostMajorPiece:
            hindsightPenaltyStrength = 0.085 * (victimMaterialValue / 9.0)
            if victimMaterialValue >= 9.0:
                hindsightPenaltyStrength *= 1.45
            blackPlySlotsFilled = 0
            walkBackIndex = transitionIndex - 1
            while walkBackIndex >= 0 and blackPlySlotsFilled < lookbackBlackPlies:
                if walkBackIndex % 2 == 1:
                    rewardAdjustmentByIndex[walkBackIndex] -= hindsightPenaltyStrength
                    blackPlySlotsFilled += 1
                walkBackIndex -= 1

    return [
        (transitionRow[0], transitionRow[1], transitionRow[2], transitionRow[3] + rewardAdjustmentByIndex[resultIndex])
        for resultIndex, transitionRow in enumerate(transitions)
    ]
