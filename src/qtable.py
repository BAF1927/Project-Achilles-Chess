# Tabular Q-learning: stores (normalized FEN key, move UCI) -> scalar Q learned from
# self-play.
# Only visited states are kept so the file stays bounded.
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class QTable:
    learningRateAlpha: float
    discountGamma: float
    entries: Dict[str, Dict[str, float]]


# Strips half-move and full-move counters so two positions that differ only by clocks
# share one key and generalization improves.
def normalizeBoardPositionKey(fenString: str) -> str:
    tokenParts = fenString.split()
    if len(tokenParts) < 4:
        return fenString
    return " ".join(tokenParts[:4])


def pathForSharedQTableFile() -> Path:
    return Path(__file__).resolve().parent / "qtable.json"


def pathForDifficultyQTableFile(difficultyLevel: str) -> Path:
    normalizedDifficulty = difficultyLevel.lower().strip()
    if normalizedDifficulty not in ("easy", "medium", "hard", "achilles"):
        normalizedDifficulty = "medium"
    return Path(__file__).resolve().parent / f"qtable_{normalizedDifficulty}.json"


def loadQTable(difficultyLevel: str | None = None) -> QTable:
    tablePath = pathForDifficultyQTableFile(difficultyLevel) if difficultyLevel else pathForSharedQTableFile()
    if not tablePath.exists():
        return QTable(learningRateAlpha=0.15, discountGamma=0.92, entries={})
    with open(tablePath, "r", encoding="utf-8") as fileHandle:
        filePayload = json.load(fileHandle)
    rawEntries = dict(filePayload.get("entries", {}))
    normalizedEntries: Dict[str, Dict[str, float]] = {}
    for fenKey, moveValueMap in rawEntries.items():
        canonicalKey = normalizeBoardPositionKey(str(fenKey))
        if canonicalKey not in normalizedEntries:
            normalizedEntries[canonicalKey] = {}
        for moveUci, rawQValue in dict(moveValueMap).items():
            previousValue = normalizedEntries[canonicalKey].get(moveUci)
            numericValue = float(rawQValue)
            normalizedEntries[canonicalKey][moveUci] = (
                numericValue if previousValue is None else max(float(previousValue), numericValue)
            )
    return QTable(
        learningRateAlpha=float(filePayload.get("alpha", 0.15)),
        discountGamma=float(filePayload.get("gamma", 0.92)),
        entries=normalizedEntries,
    )


def saveQTable(table: QTable, difficultyLevel: str | None = None) -> None:
    filePayload = {
        "version": 1,
        "alpha": table.learningRateAlpha,
        "gamma": table.discountGamma,
        "entries": table.entries,
    }
    tablePath = pathForDifficultyQTableFile(difficultyLevel) if difficultyLevel else pathForSharedQTableFile()
    with open(tablePath, "w", encoding="utf-8") as fileHandle:
        json.dump(filePayload, fileHandle, indent=2)
        fileHandle.write("\n")


def readQValue(table: QTable, fenString: str, moveUci: str) -> float | None:
    canonicalKey = normalizeBoardPositionKey(fenString)
    moveDictionary = table.entries.get(canonicalKey)
    if not moveDictionary:
        return None
    if moveUci not in moveDictionary:
        return None
    return float(moveDictionary[moveUci])


def writeQValue(table: QTable, fenString: str, moveUci: str, newValue: float) -> None:
    canonicalKey = normalizeBoardPositionKey(fenString)
    if canonicalKey not in table.entries:
        table.entries[canonicalKey] = {}
    table.entries[canonicalKey][moveUci] = float(newValue)


def maximumQValueAtPosition(table: QTable, nextPositionKey: str) -> float:
    moveDictionary = table.entries.get(nextPositionKey)
    if not moveDictionary:
        return 0.0
    return max(float(value) for value in moveDictionary.values()) if moveDictionary else 0.0


def applyBellmanBackup(
    table: QTable,
    fenString: str,
    moveUci: str,
    immediateReward: float,
    nextPositionFen: str,
) -> float:
    # Bellman / one-step TD: nudge Q(state, action) toward "immediate reward plus discounted
    # best value from the next position" (good moves lead to good follow-ups). Self-play
    # training calls this; at runtime the table feeds PyReason as optional QValue facts.
    # Q(s,a) <- (1-alpha)*Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a'))
    previousEstimate = readQValue(table, fenString, moveUci)
    if previousEstimate is None:
        previousEstimate = 0.0
    nextKey = normalizeBoardPositionKey(nextPositionFen)
    bootstrapTarget = float(immediateReward) + table.discountGamma * maximumQValueAtPosition(table, nextKey)
    updatedValue = (1.0 - table.learningRateAlpha) * float(previousEstimate) + table.learningRateAlpha * bootstrapTarget
    writeQValue(table, fenString, moveUci, updatedValue)
    return updatedValue


# Maps a real-valued Q score to a tight [0,1] interval suitable as a PyReason fact
# bound.
def qLearningValueToPyreasonBounds(unboundedQValue: float) -> Tuple[float, float]:
    import math

    centerUnit = 1.0 / (1.0 + math.exp(-unboundedQValue))
    centerUnit = max(0.01, min(0.99, centerUnit))
    lowBound = max(0.01, centerUnit - 0.02)
    highBound = min(0.99, centerUnit + 0.02)
    return lowBound, highBound


# Stable names used across trainers (thin aliases matching JSON field names on disk).
def normalizeFenKey(fenString: str) -> str:
    return normalizeBoardPositionKey(fenString)


def getQValue(table: QTable, fenString: str, moveUci: str) -> float | None:
    return readQValue(table, fenString, moveUci)


def setQValue(table: QTable, fenString: str, moveUci: str, value: float) -> None:
    writeQValue(table, fenString, moveUci, value)


def maxNextQ(table: QTable, nextFen: str) -> float:
    return maximumQValueAtPosition(table, normalizeBoardPositionKey(nextFen))


def updateQ(table: QTable, fen: str, uci: str, reward: float, nextFen: str) -> float:
    return applyBellmanBackup(table, fen, uci, reward, nextFen)


def qToBound(qValue: float) -> Tuple[float, float]:
    return qLearningValueToPyreasonBounds(qValue)
