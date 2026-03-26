# Board rules, piece sprites, and move animation state.
# Move selection for Black uses PyReasonChessAI; this layer only applies moves the user or
# AI chose.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import chess
import pygame

from .chess_ai import PyReasonChessAI
from .config import boardSize, moveAnimationDurationMilliseconds, piecesAssetsDirectory, squareSizePixels

pieceTypeDisplayNames = {
    chess.PAWN: "Pawn",
    chess.KNIGHT: "Knight",
    chess.BISHOP: "Bishop",
    chess.ROOK: "Rook",
    chess.QUEEN: "Queen",
    chess.KING: "King",
}

pieceImageFilenameByColorAndType = {
    (chess.WHITE, chess.PAWN): "white-pawn.png",
    (chess.WHITE, chess.KNIGHT): "white-knight.png",
    (chess.WHITE, chess.BISHOP): "white-bishop.png",
    (chess.WHITE, chess.ROOK): "white-rook.png",
    (chess.WHITE, chess.QUEEN): "white-queen.png",
    (chess.WHITE, chess.KING): "white-king.png",
    (chess.BLACK, chess.PAWN): "black-pawn.png",
    (chess.BLACK, chess.KNIGHT): "black-knight.png",
    (chess.BLACK, chess.BISHOP): "black-bishop.png",
    (chess.BLACK, chess.ROOK): "black-rook.png",
    (chess.BLACK, chess.QUEEN): "black-queen.png",
    (chess.BLACK, chess.KING): "black-king.png",
}


@dataclass
class MoveAnimation:
    fromSquare: int
    toSquare: int
    pieceColor: chess.Color
    pieceType: chess.PieceType
    startTimeMilliseconds: int
    durationMilliseconds: int = moveAnimationDurationMilliseconds


@dataclass
class GameState:
    board: chess.Board = field(default_factory=chess.Board)
    selectedSquare: Optional[int] = None
    selectedMoves: List[chess.Move] = field(default_factory=list)
    aiReasons: List[str] = field(default_factory=list)
    aiThinking: bool = False
    activeAnimation: Optional[MoveAnimation] = None


# Locate the mated king and list enemy units attacking that square for overlays and status
# text.
def describeCheckmateAttackingPieces(board: chess.Board) -> str:
    if not board.is_checkmate():
        return ""
    kingSquare = board.king(board.turn)
    if kingSquare is None:
        return ""
    attackerSquares = board.attackers(not board.turn, kingSquare)
    descriptionParts: List[str] = []
    for attackerSquare in sorted(attackerSquares):
        attackingPiece = board.piece_at(attackerSquare)
        if attackingPiece is None:
            continue
        pieceLabel = pieceTypeDisplayNames.get(attackingPiece.piece_type, "Piece")
        descriptionParts.append(f"{pieceLabel} on {chess.square_name(attackerSquare)}")
    if not descriptionParts:
        return "Unknown attacker"
    if len(descriptionParts) == 1:
        return descriptionParts[0]
    return ", ".join(descriptionParts)


class ChessGame:
    def __init__(self, difficulty: str = "medium") -> None:
        self.difficulty = difficulty
        self.state = GameState()
        self.ai = PyReasonChessAI(difficulty=difficulty)
        self.pieceImages = self.loadPieceImages()

    def loadPieceImages(self) -> Dict[tuple, pygame.Surface]:
        if not piecesAssetsDirectory.exists():
            raise FileNotFoundError(f"Missing pieces folder: {piecesAssetsDirectory}")
        imagesByPieceKey: Dict[tuple, pygame.Surface] = {}
        for colorAndType, filename in pieceImageFilenameByColorAndType.items():
            imagePath = piecesAssetsDirectory / filename
            if not imagePath.exists():
                raise FileNotFoundError(f"Missing sprite: {imagePath}")
            surface = pygame.image.load(str(imagePath))
            if pygame.display.get_surface() is not None:
                surface = surface.convert_alpha()
            imagesByPieceKey[colorAndType] = pygame.transform.smoothscale(
                surface, (squareSizePixels, squareSizePixels)
            )
        return imagesByPieceKey

    def resetBoardToNewGame(self) -> None:
        self.state = GameState()

    def isAnimatingMove(self) -> bool:
        return self.state.activeAnimation is not None

    def animationProgress(self, currentTimeMilliseconds: int) -> Tuple[float, Optional[MoveAnimation]]:
        activeAnimation = self.state.activeAnimation
        if activeAnimation is None:
            return 1.0, None
        elapsedMilliseconds = currentTimeMilliseconds - activeAnimation.startTimeMilliseconds
        progressUnitInterval = min(
            1.0, elapsedMilliseconds / float(max(1, activeAnimation.durationMilliseconds))
        )
        return progressUnitInterval, activeAnimation

    def updateAnimation(self, currentTimeMilliseconds: int) -> None:
        activeAnimation = self.state.activeAnimation
        if activeAnimation is None:
            return
        if currentTimeMilliseconds - activeAnimation.startTimeMilliseconds >= activeAnimation.durationMilliseconds:
            self.state.activeAnimation = None

    def recordMoveAndStartAnimation(self, move: chess.Move, currentTimeMilliseconds: int) -> None:
        board = self.state.board
        board.push(move)
        movedPiece = board.piece_at(move.to_square)
        if movedPiece is None:
            return
        self.state.activeAnimation = MoveAnimation(
            fromSquare=move.from_square,
            toSquare=move.to_square,
            pieceColor=movedPiece.color,
            pieceType=movedPiece.piece_type,
            startTimeMilliseconds=currentTimeMilliseconds,
        )

    @staticmethod
    def squareIndexFromScreenGrid(fileIndex: int, rankIndexFromTop: int) -> int:
        # Screen rank 0 is the top row (Black's back rank in standard setup).
        return chess.square(fileIndex, boardSize - 1 - rankIndexFromTop)

    def selectSquare(self, squareIndex: int) -> None:
        board = self.state.board
        self.state.selectedSquare = None
        self.state.selectedMoves = []
        piece = board.piece_at(squareIndex)
        if piece is None:
            return
        if board.turn != chess.WHITE:
            return
        if piece.color != chess.WHITE:
            return
        self.state.selectedSquare = squareIndex
        self.state.selectedMoves = [legalMove for legalMove in board.legal_moves if legalMove.from_square == squareIndex]

    def handlePlayerClick(self, fileIndex: int, rankIndexFromTop: int, currentTimeMilliseconds: int) -> bool:
        board = self.state.board
        if self.state.activeAnimation is not None:
            return False
        if board.is_game_over() or board.turn != chess.WHITE:
            return False
        clickedSquare = self.squareIndexFromScreenGrid(fileIndex, rankIndexFromTop)
        if self.state.selectedSquare is None:
            self.selectSquare(clickedSquare)
            return False
        for candidateMove in self.state.selectedMoves:
            if candidateMove.to_square == clickedSquare:
                self.state.selectedSquare = None
                self.state.selectedMoves = []
                self.recordMoveAndStartAnimation(candidateMove, currentTimeMilliseconds)
                return True
        self.selectSquare(clickedSquare)
        return False

    # The play loop calls this every frame; it only runs the AI when it is Black's turn, the
    # game is still going, and no move animation is active (so it is safe to act).
    def moveOnlyWhenTurn(self, currentTimeMilliseconds: int) -> bool:
        board = self.state.board
        if self.state.activeAnimation is not None:
            return False
        if board.is_game_over() or board.turn != chess.BLACK:
            return False
        self.state.aiThinking = True
        chosenMove, explanationLines = self.ai.chooseMove(board)
        self.recordMoveAndStartAnimation(chosenMove, currentTimeMilliseconds)
        self.state.aiReasons = explanationLines
        self.state.aiThinking = False
        return True
