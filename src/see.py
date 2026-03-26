# Static exchange evaluation (SEE) pruning and anti-shuffle helpers.
# Non-king moves that land on a square where the opponent wins material in a forcing
# capture sequence are dropped.
from __future__ import annotations

import chess

from .features import pieceValuesByType


# Net material (pawn units) the side to move can expect from captures on targetSquare,
# assuming both sides always recapture on that square with the cheapest correct attacker.
def seeGainForSideToMove(board: chess.Board, targetSquare: chess.Square) -> float:
    captureMovesToSquare = [
        legalMove
        for legalMove in board.legal_moves
        if legalMove.to_square == targetSquare
        and board.piece_at(targetSquare) is not None
        and board.piece_at(legalMove.from_square) is not None
    ]
    if not captureMovesToSquare:
        return 0.0
    victimPiece = board.piece_at(targetSquare)
    if victimPiece is None:
        return 0.0
    victimMaterialValue = float(pieceValuesByType[victimPiece.piece_type])
    chosenCapture = min(
        captureMovesToSquare,
        key=lambda captureMove: (
            pieceValuesByType[board.piece_at(captureMove.from_square).piece_type],
            captureMove.from_square,
            captureMove.to_square,
        ),
    )
    board.push(chosenCapture)
    recursiveGain = victimMaterialValue - seeGainForSideToMove(board, targetSquare)
    board.pop()
    return max(0.0, recursiveGain)


# True if the opponent's SEE score on the landing square exceeds what we are willing to
# lose: for captures, the threshold is reduced by the material we already banked this move.
def moveIsSeeTacticalBlunder(board: chess.Board, move: chess.Move) -> bool:
    movingPiece = board.piece_at(move.from_square)
    if movingPiece is None or movingPiece.piece_type == chess.KING:
        return False

    capturedMaterialBanked = 0.0
    if board.is_en_passant(move):
        capturedMaterialBanked = float(pieceValuesByType[chess.PAWN])
    elif board.is_capture(move):
        capturedVictim = board.piece_at(move.to_square)
        if capturedVictim is not None:
            capturedMaterialBanked = float(pieceValuesByType[capturedVictim.piece_type])

    board.push(move)
    arrivalSquare = move.to_square
    pieceAfterMove = board.piece_at(arrivalSquare)
    if pieceAfterMove is None or pieceAfterMove.color != (not board.turn):
        board.pop()
        return False
    opponentSeeGain = seeGainForSideToMove(board, arrivalSquare)
    board.pop()

    if capturedMaterialBanked > 0.0:
        return opponentSeeGain > capturedMaterialBanked + 0.15
    return opponentSeeGain > 0.0


def isImmediateUndoOfOwnLastMove(board: chess.Board, move: chess.Move) -> bool:
    if len(board.move_stack) < 2:
        return False
    ownPreviousMove = board.move_stack[-2]
    return move.from_square == ownPreviousMove.to_square and move.to_square == ownPreviousMove.from_square


def filterOutShuffleMoves(board: chess.Board, legalMoves: list[chess.Move]) -> list[chess.Move]:
    keptMoves = [legalMove for legalMove in legalMoves if not isImmediateUndoOfOwnLastMove(board, legalMove)]
    return keptMoves if keptMoves else legalMoves


def filterOutSeeLosingMoves(board: chess.Board, legalMoves: list[chess.Move]) -> list[chess.Move]:
    keptMoves = [legalMove for legalMove in legalMoves if not moveIsSeeTacticalBlunder(board, legalMove)]
    return keptMoves if keptMoves else legalMoves
