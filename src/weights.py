# Loads and saves per-difficulty linear weights for turning MoveFeatures into PyReason facts.
# Shipped JSON keeps the game playable without running the evolutionary trainer first.
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class Weights:
    bias: float
    captureValue: float
    givesCheck: float
    isCenter: float
    hangsMovedPiece: float
    winsMaterialAfterExchange: float
    pinnedMovedPiece: float
    kingPressure: float
    crampOpponent: float
    developsOffBackRank: float
    underFireQuality: float


def trainedWeightsJsonPath() -> Path:
    return Path(__file__).resolve().parent / "trainedWeights.json"


def loadWeightsByDifficulty() -> Dict[str, Weights]:
    path = trainedWeightsJsonPath()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fileHandle:
        payload = json.load(fileHandle)
    result: Dict[str, Weights] = {}
    for difficultyKey in ("easy", "medium", "hard", "achilles"):
        if difficultyKey not in payload:
            continue
        row = payload[difficultyKey]
        result[difficultyKey] = Weights(
            bias=float(row.get("bias", 0.0)),
            captureValue=float(row.get("captureValue", 0.0)),
            givesCheck=float(row.get("givesCheck", 0.0)),
            isCenter=float(row.get("isCenter", 0.0)),
            hangsMovedPiece=float(row.get("hangsMovedPiece", 0.0)),
            winsMaterialAfterExchange=float(row.get("winsMaterialAfterExchange", 0.0)),
            pinnedMovedPiece=float(row.get("pinnedMovedPiece", -1.1)),
            kingPressure=float(row.get("kingPressure", 0.85)),
            crampOpponent=float(row.get("crampOpponent", 0.55)),
            developsOffBackRank=float(row.get("developsOffBackRank", 0.45)),
            underFireQuality=float(row.get("underFireQuality", 1.0)),
        )
    return result


def saveWeightsByDifficulty(weightsByDifficulty: Dict[str, Weights]) -> None:
    path = trainedWeightsJsonPath()
    payload: Dict[str, object] = {
        "version": 1,
        "features": [
            "captureValue",
            "givesCheck",
            "isCenter",
            "hangsMovedPiece",
            "winsMaterialAfterExchange",
            "pinnedMovedPiece",
            "kingPressure",
            "crampOpponent",
            "developsOffBackRank",
            "underFireQuality",
        ],
    }
    for difficultyKey, weightVector in weightsByDifficulty.items():
        payload[difficultyKey] = {
            "bias": weightVector.bias,
            "captureValue": weightVector.captureValue,
            "givesCheck": weightVector.givesCheck,
            "isCenter": weightVector.isCenter,
            "hangsMovedPiece": weightVector.hangsMovedPiece,
            "winsMaterialAfterExchange": weightVector.winsMaterialAfterExchange,
            "pinnedMovedPiece": weightVector.pinnedMovedPiece,
            "kingPressure": weightVector.kingPressure,
            "crampOpponent": weightVector.crampOpponent,
            "developsOffBackRank": weightVector.developsOffBackRank,
            "underFireQuality": weightVector.underFireQuality,
        }
    with open(path, "w", encoding="utf-8") as fileHandle:
        json.dump(payload, fileHandle, indent=2)
        fileHandle.write("\n")
