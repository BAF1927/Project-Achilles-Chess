# PyReason-driven move selection: legal moves become graph nodes with interval-valued
# facts; rules lift them into Ranked(moves) scores and Python picks the strongest lower
# bound.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import chess

from .features import MoveFeatures, computeMoveFeatures
from .pyreason_move_selection import (
    bootstrapPyreason,
    pickUciFromPyreasonScores,
    runPyreasonForMoves,
)
from .weights import Weights, loadWeightsByDifficulty
from .qtable import QTable, getQValue, loadQTable, normalizeFenKey, qToBound
from .see import filterOutSeeLosingMoves, filterOutShuffleMoves
from .training_shaping import filterOutPieceShuttleMoves

weightsLoadedByDifficulty = loadWeightsByDifficulty()


@dataclass
class MoveScore:
    # Debug bundle; tie-breaking always comes from PyReason bounds, not this struct.
    move: chess.Move
    score: float
    reasons: List[str]


class PyReasonChessAI:
    def __init__(
        self,
        difficulty: str = "medium",
        rngSeed: int = 42,
    ) -> None:
        allowedDifficultyKeys = frozenset({"easy", "medium", "hard", "achilles"})
        self.difficulty = difficulty if difficulty in allowedDifficultyKeys else "medium"
        self.weights = (
            weightsLoadedByDifficulty.get(self.difficulty)
            or weightsLoadedByDifficulty.get("hard")
            or Weights(
                bias=0.0,
                captureValue=1.6,
                givesCheck=1.0,
                isCenter=0.5,
                hangsMovedPiece=-2.0,
                winsMaterialAfterExchange=0.8,
                pinnedMovedPiece=-1.1,
                kingPressure=0.85,
                crampOpponent=0.55,
                developsOffBackRank=0.45,
                underFireQuality=1.0,
            )
        )
        try:
            import pyreason as pyreasonModule  # noqa: PLC0415
        except ImportError as importError:
            raise SystemExit(
                "PyReason must be installed for this project.\n"
                "pip install pyreason  (use Python 3.9/3.10)\n"
                "https://pyreason.readthedocs.io/en/latest/installation.html"
            ) from importError
        self.pyreasonModule = pyreasonModule
        bootstrapPyreason(pyreasonModule)
        self.qtable: QTable = loadQTable(self.difficulty)

    def mapToUnitInterval(self, value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def featureToReasoningBounds(self, weight: float, featureUnit: float, invert: bool = False) -> Tuple[float, float]:
        # Turns a weighted feature in [0,1] into a fuzzy interval for PyReason.
        # Positive weights increase how much high feature values pull the interval upward.
        effectiveFeature = featureUnit if not invert else (1.0 - featureUnit)
        logits = self.weights.bias + weight * effectiveFeature
        center = max(0.01, min(0.99, self.mapToUnitInterval(logits)))
        lowBound = max(0.01, center - 0.02)
        highBound = min(0.99, center + 0.02)
        return lowBound, highBound

    def buildPredicateFacts(self, features: MoveFeatures) -> Dict[str, Tuple[float, float]]:
        # Maps each PyReason predicate name to a numeric bound pair for one candidate move.
        return {
            # Captures a piece; strength scales with victim material (pawn … queen).
            "Capture": self.featureToReasoningBounds(self.weights.captureValue, features.captureValue),
            # This move gives check to the opponent king.
            "Check": self.featureToReasoningBounds(self.weights.givesCheck, features.givesCheck),
            # Destination is a central square (d4, e4, d5, e5).
            "Center": self.featureToReasoningBounds(self.weights.isCenter, features.isCenter),
            # Hanging is bad, so invert so the Safe predicate rises when the piece is not
            # hanging.
            "Safe": self.featureToReasoningBounds(self.weights.hangsMovedPiece, features.hangsMovedPiece, invert=True),
            # Forcing recapture sequence on this square wins material for us.
            "TradeUp": self.featureToReasoningBounds(
                self.weights.winsMaterialAfterExchange, features.winsMaterialAfterExchange
            ),
            # Bad to move the piece onto a pinned square; invert so PinOk rises when it is
            # not pinned after the move.
            "PinOk": self.featureToReasoningBounds(self.weights.pinnedMovedPiece, features.movedPiecePinned, invert=True),
            # After the move, we attack squares around the enemy king.
            "KingPress": self.featureToReasoningBounds(self.weights.kingPressure, features.kingNeighborhoodPressure),
            # After the move, the opponent has fewer legal moves (more cramped).
            "Cramp": self.featureToReasoningBounds(self.weights.crampOpponent, features.crampOpponent),
            # Piece leaves the back rank (development), not counting king/pawn quirks in the
            # feature extractor.
            "Develop": self.featureToReasoningBounds(self.weights.developsOffBackRank, features.developsOffBackRank),
            # Queen/rook on an attacked square: quality of this reply (retreat, take
            # attacker, or best loss). Irrelevant pieces get a neutral feature (0.5).
            "UnderFire": self.featureToReasoningBounds(self.weights.underFireQuality, features.underFireQuality),
        }

    def chooseMove(self, board: chess.Board) -> Tuple[chess.Move, List[str]]:
        legalMoves = list(board.legal_moves)
        if not legalMoves:
            raise RuntimeError("No legal moves")
        legalMoves = filterOutSeeLosingMoves(board, legalMoves)
        legalMoves = filterOutShuffleMoves(board, legalMoves)
        legalMoves = filterOutPieceShuttleMoves(board, legalMoves)

        uciToPredicateFacts: Dict[str, Dict[str, Tuple[float, float]]] = {}
        positionKey = normalizeFenKey(board.fen())
        for move in legalMoves:
            moveFeatures = computeMoveFeatures(board, move)
            predicateFacts = self.buildPredicateFacts(moveFeatures)
            storedQ = getQValue(self.qtable, positionKey, move.uci())
            if storedQ is not None:
                predicateFacts["QValue"] = qToBound(storedQ)
            uciToPredicateFacts[move.uci()] = predicateFacts

        scoresByUci, inferenceNote = runPyreasonForMoves(self.pyreasonModule, legalMoves, uciToPredicateFacts)
        bestUci = pickUciFromPyreasonScores(scoresByUci)
        if bestUci is None:
            raise RuntimeError("PyReason returned no scored moves—check PyReason install and rules.")

        chosenMove = chess.Move.from_uci(bestUci)
        explanationLines: List[str] = [
            "move_choice=argmax(PyReason Ranked lower bounds)",
            inferenceNote,
            "see_prefilter=see_all_moves+shuffle+shuttle_trim",
            f"difficulty={self.difficulty}",
        ]

        return chosenMove, explanationLines
