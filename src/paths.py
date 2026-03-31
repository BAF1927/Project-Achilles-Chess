# All filesystem locations for this assessment package live here so other modules do not hard-code folders.
# data/: bundled rules.json and yago_subset.graphml. output/: normal write target for inference and LLM files.
# brunoOutput/: optional checked-in snapshot; the dashboard reads it when env LEIBNIZ_USE_BRUNO_OUTPUT is set.
# filterShowableRules drops rules.json entries with "showable": false from OpenAI, PyReason export, and UI.
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

assessmentPackageDirectory = Path(__file__).resolve().parent
repositoryRootDirectory = assessmentPackageDirectory.parent
dataDirectory = assessmentPackageDirectory / "data"
# Live OpenAI + PyReason writes always go here (never the frozen sample tree).
writableOutputDirectory = assessmentPackageDirectory / "output"
# Checked-in snapshot for demos; dashboard reads this when ``LEIBNIZ_USE_BRUNO_OUTPUT=1``.
brunoSampleOutputDirectory = assessmentPackageDirectory / "brunoOutput"
envExampleFileName = ".env.example"
graphmlFileName = "yago_subset.graphml"
rulesFileName = "rules.json"
inferenceResultFileName = "inference_result.json"
openAiTranscriptFileName = "llm_openai_prompts_and_replies.json"


def graphmlPath() -> Path:
    # Single KG file PyReason loads (YAGO subset in GraphML form).
    return dataDirectory / graphmlFileName


def rulesPath() -> Path:
    # PyReason logic rules plus UI metadata (display_title, showable, description, ...).
    return dataDirectory / rulesFileName


def artifactDirectory(*, write: bool = False) -> Path:
    # If write is True, return writableOutputDirectory only. If False, may return brunoSampleOutputDirectory for demos.
    if write:
        return writableOutputDirectory
    if os.environ.get("LEIBNIZ_USE_BRUNO_OUTPUT", "").strip().lower() in ("1", "true", "yes"):
        return brunoSampleOutputDirectory
    return writableOutputDirectory


def inferenceResultPath(*, write: bool = False) -> Path:
    # Bundle JSON written after reason(): inferred edges, trace sample, relative paths.
    return artifactDirectory(write=write) / inferenceResultFileName


def openAiTranscriptPath(*, write: bool = False) -> Path:
    # Saved prompts and raw assistant replies per rule (for auditing the LLM step).
    return artifactDirectory(write=write) / openAiTranscriptFileName


def repositoryEnvFilePath() -> Path:
    # Expected next to README.md (repo root), not inside src/.
    return repositoryRootDirectory / ".env"


def repositoryEnvExamplePath() -> Path:
    return repositoryRootDirectory / envExampleFileName


def filterShowableRules(rules: object) -> list[dict[str, Any]]:
    # Excludes rules with "showable": false from inference, LLM, and dashboard.
    if not isinstance(rules, list):
        return []
    return [
        r
        for r in rules
        if isinstance(r, dict) and r.get("showable", True) is not False
    ]
