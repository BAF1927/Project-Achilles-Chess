# Maps a legal chess move from a position into numeric features in [0, 1].
# The same extractor runs at training time and at runtime so PyReason always sees consistent
# signals.
from __future__ import annotations

from dataclasses import dataclass

import chess


pieceValuesByType = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

centerSquareSet = {chess.D4, chess.E4, chess.D5, chess.E5}


@dataclass(frozen=True)
class MoveFeatures:
    captureValue: float
    givesCheck: float
    isCenter: float
    hangsMovedPiece: float
    winsMaterialAfterExchange: float
    movedPiecePinned: float
    kingNeighborhoodPressure: float
    crampOpponent: float
    developsOffBackRank: float
    underFireQuality: float


def clampToUnitInterval(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


# Counts how many squares around the enemy king are attacked by us; higher means more king
# pressure.
def kingNeighborhoodPressureScore(board: chess.Board, friendlyColor: chess.Color) -> float:
    enemyColor = not friendlyColor
    enemyKingSquare = board.king(enemyColor)
    if enemyKingSquare is None:
        return 0.0
    kingFileIndex, kingRankIndex = chess.square_file(enemyKingSquare), chess.square_rank(enemyKingSquare)
    attackedNeighborCount = 0
    neighborSquareCount = 0
    for deltaFile in (-1, 0, 1):
        for deltaRank in (-1, 0, 1):
            if deltaFile == 0 and deltaRank == 0:
                continue
            neighborFile = kingFileIndex + deltaFile
            neighborRank = kingRankIndex + deltaRank
            if 0 <= neighborFile < 8 and 0 <= neighborRank < 8:
                neighborSquareCount += 1
                neighborSquareIndex = chess.square(neighborFile, neighborRank)
                if board.is_attacked_by(friendlyColor, neighborSquareIndex):
                    attackedNeighborCount += 1
    return clampToUnitInterval(attackedNeighborCount / float(max(neighborSquareCount, 1)))


def bestCaptureValueFromSquare(board: chess.Board, originSquare: chess.Square) -> int:
    bestValue = 0
    for legalMove in board.legal_moves:
        if legalMove.from_square != originSquare:
            continue
        if board.is_en_passant(legalMove):
            bestValue = max(bestValue, 1)
        elif board.is_capture(legalMove):
            victim = board.piece_at(legalMove.to_square)
            if victim is not None:
                bestValue = max(bestValue, pieceValuesByType[victim.piece_type])
    return int(bestValue)


def captureMaterialValueOfMove(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return 1
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        return int(pieceValuesByType[victim.piece_type]) if victim is not None else 0
    return 0


def everyMoveFromSquareFailsSee(board: chess.Board, originSquare: chess.Square) -> bool:
    from .see import moveIsSeeTacticalBlunder

    movesFromSquare = [legalMove for legalMove in board.legal_moves if legalMove.from_square == originSquare]
    if not movesFromSquare:
        return False
    return all(moveIsSeeTacticalBlunder(board, legalMove) for legalMove in movesFromSquare)


def underFireMoveQualityScore(board: chess.Board, move: chess.Move, movingPiece: chess.Piece | None) -> float:
    # When a queen or rook stands on an attacked square, scores retreats, capturing an
    # attacker, or (if every SEE-legal option loses) the strongest desperado capture.
    if movingPiece is None or movingPiece.piece_type not in (chess.QUEEN, chess.ROOK):
        return 0.5

    friendlyColor = movingPiece.color
    enemyColor = not friendlyColor
    originSquare = move.from_square
    enemyAttackersOnOrigin = board.attackers(enemyColor, originSquare)
    if not enemyAttackersOnOrigin:
        return 0.5

    enemyAttackersOnDestination = board.attackers(enemyColor, move.to_square)
    attackerCountOnOrigin = len(enemyAttackersOnOrigin)
    attackerCountOnDestination = len(enemyAttackersOnDestination)

    if everyMoveFromSquareFailsSee(board, originSquare):
        bestCaptureValue = bestCaptureValueFromSquare(board, originSquare)
        thisCaptureValue = captureMaterialValueOfMove(board, move)
        if bestCaptureValue == 0:
            return 0.18
        if thisCaptureValue >= bestCaptureValue:
            return clampToUnitInterval(0.72 + 0.28 * (thisCaptureValue / 9.0))
        return 0.14

    victimOnDestination = board.piece_at(move.to_square)
    if (board.is_capture(move) or board.is_en_passant(move)) and move.to_square in enemyAttackersOnOrigin:
        if victimOnDestination is not None:
            victimValue = pieceValuesByType[victimOnDestination.piece_type]
            return clampToUnitInterval(0.58 + 0.42 * (victimValue / 9.0))

    if not enemyAttackersOnDestination:
        return 1.0

    if attackerCountOnDestination < attackerCountOnOrigin:
        return clampToUnitInterval(0.62 + 0.38 * (1.0 - attackerCountOnDestination / max(attackerCountOnOrigin, 1)))

    if board.is_capture(move) or board.is_en_passant(move):
        if victimOnDestination is not None:
            return clampToUnitInterval(0.35 + 0.3 * (pieceValuesByType[victimOnDestination.piece_type] / 9.0))
    return 0.2


def developsOffBackRankScore(board: chess.Board, move: chess.Move, movingPiece: chess.Piece | None) -> float:
    if movingPiece is None or movingPiece.piece_type in (chess.KING, chess.PAWN):
        return 0.0
    homeRankIndex = 0 if movingPiece.color == chess.WHITE else 7
    if chess.square_rank(move.from_square) != homeRankIndex:
        return 0.0
    if chess.square_rank(move.to_square) == homeRankIndex:
        return 0.0
    return 1.0


def computeMoveFeatures(board: chess.Board, move: chess.Move) -> MoveFeatures:
    movingPiece = board.piece_at(move.from_square)
    destinationOccupant = board.piece_at(move.to_square)

    underFireQuality = underFireMoveQualityScore(board, move, movingPiece)

    captureValueFeature = 0.0
    if destinationOccupant is not None:
        captureValueFeature = clampToUnitInterval(pieceValuesByType[destinationOccupant.piece_type] / 9.0)

    centerControlFeature = 1.0 if move.to_square in centerSquareSet else 0.0

    board.push(move)
    checkFeature = 1.0 if board.is_check() else 0.0

    hangsMovedPieceFeature = 0.0
    if movingPiece is not None:
        enemyAttackers = board.attackers(not movingPiece.color, move.to_square)
        friendlyDefenders = board.attackers(movingPiece.color, move.to_square)
        if enemyAttackers and not friendlyDefenders:
            hangsMovedPieceFeature = 1.0
        elif (
            movingPiece.piece_type == chess.QUEEN
            and enemyAttackers
            and len(enemyAttackers) > len(friendlyDefenders)
        ):
            hangsMovedPieceFeature = max(hangsMovedPieceFeature, 0.7)

    winsMaterialAfterExchangeFeature = 0.0
    if movingPiece is not None:
        opponentColor = not movingPiece.color
        opponentCapturers = list(board.attackers(opponentColor, move.to_square))
        if opponentCapturers:
            cheapestCapturerSquare = min(
                opponentCapturers,
                key=lambda square: pieceValuesByType[board.piece_at(square).piece_type]
                if board.piece_at(square)
                else 99,
            )
            capturerPiece = board.piece_at(cheapestCapturerSquare)
            if capturerPiece is not None:
                materialGainIfOpponentTakes = pieceValuesByType[movingPiece.piece_type]
                materialCostForOpponent = pieceValuesByType[capturerPiece.piece_type]
                if materialCostForOpponent > materialGainIfOpponentTakes:
                    winsMaterialAfterExchangeFeature = 1.0

    movedPiecePinnedFeature = 0.0
    if movingPiece is not None and movingPiece.piece_type != chess.KING:
        if board.is_pinned(movingPiece.color, move.to_square):
            movedPiecePinnedFeature = 1.0

    kingPressureFeature = 0.0
    crampOpponentFeature = 0.0
    if movingPiece is not None:
        kingPressureFeature = kingNeighborhoodPressureScore(board, movingPiece.color)
        opponentLegalMoveCount = len(list(board.legal_moves))
        crampOpponentFeature = clampToUnitInterval(1.0 - opponentLegalMoveCount / 38.0)

    developsFeature = developsOffBackRankScore(board, move, movingPiece)

    board.pop()

    return MoveFeatures(
        captureValue=captureValueFeature,
        givesCheck=checkFeature,
        isCenter=centerControlFeature,
        hangsMovedPiece=hangsMovedPieceFeature,
        winsMaterialAfterExchange=winsMaterialAfterExchangeFeature,
        movedPiecePinned=movedPiecePinnedFeature,
        kingNeighborhoodPressure=kingPressureFeature,
        crampOpponent=crampOpponentFeature,
        developsOffBackRank=developsFeature,
        underFireQuality=underFireQuality,
    )
