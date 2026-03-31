# HTTP client for OpenAI Chat Completions. Each showable rule gets one JSON response with facts for PyReason.
# resolveLlmPayload calls buildPayloadWithOpenAi; fact strings are cleaned so they end correctly for PyReason.
# Env vars: OPENAI_API_KEY, OPENAI_MODEL, LLM_FACTS_PER_RULE, LLM_NATURAL_LANGUAGE_EXAMPLES_PER_RULE.
from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
import certifi
import networkx as nx
from . import paths


def loadRepositoryEnvironment() -> None:
    # Load .env at repo root first, then default dotenv behavior (API key for runInference).
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    envPath = paths.repositoryEnvFilePath()
    if envPath.is_file():
        load_dotenv(envPath, override=False)
    # Second load: current working directory, without overriding keys already set from repo .env.
    load_dotenv()


def readOpenAiApiKeyFromEnvironment() -> str:
    # Strip BOM and accidental quotes some editors add around the key.
    rawValue = os.environ.get("OPENAI_API_KEY", "")
    return rawValue.strip().lstrip("\ufeff").strip('"\'')


def tryFetchOpenAiChatJson(
    userPromptText: str, modelName: str = "gpt-4o-mini"
) -> tuple[dict[str, Any] | None, str | None]:
    # POST to chat/completions; retry without response_format if the model rejects json_object mode.
    apiKey = readOpenAiApiKeyFromEnvironment()
    if not apiKey:
        return None, "OPENAI_API_KEY is empty. Put it in .env at the repo root (same folder as README.md) or export it in the shell."

    sslContext = ssl.create_default_context(cafile=certifi.where())
    # Prefer JSON object mode + room for 5 NL lines + 5 facts (avoids truncated / broken strings).
    attempts: list[dict[str, Any]] = [
        {
            "model": modelName,
            "messages": [{"role": "user", "content": userPromptText}],
            "temperature": 0.1,
            "max_tokens": 8192,
            "response_format": {"type": "json_object"},
        },
        {
            "model": modelName,
            "messages": [{"role": "user", "content": userPromptText}],
            "temperature": 0.1,
            "max_tokens": 8192,
        },
    ]

    lastHttpError: str | None = None
    for payload in attempts:
        requestBody = json.dumps(payload).encode("utf-8")
        httpRequest = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=requestBody,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {apiKey}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(httpRequest, timeout=120, context=sslContext) as response:
                rawText = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            errorBody = exc.read().decode("utf-8", errors="replace")
            lastHttpError = f"OpenAI HTTP {exc.code}: {errorBody[:2000]}"
            if exc.code == 400 and "response_format" in payload and (
                "response_format" in errorBody.lower() or "json_object" in errorBody.lower()
            ):
                continue
            return None, lastHttpError
        except urllib.error.URLError as exc:
            return None, f"OpenAI network error: {exc.reason!s}"
        except TimeoutError:
            return None, "OpenAI request timed out after 120s."
        except OSError as exc:
            return None, f"OpenAI connection error: {exc!s}"

        try:
            return json.loads(rawText), None
        except json.JSONDecodeError as exc:
            snippet = rawText[:800] if rawText else ""
            return None, f"OpenAI returned non-JSON: {exc!s}. Body starts with: {snippet!r}"

    return None, lastHttpError or "OpenAI request failed."


def extractAssistantTextFromChatCompletion(responseJson: dict[str, Any]) -> str | None:
    # OpenAI response shape: choices[0].message.content string.
    choices = responseJson.get("choices")
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    return None


def parseJsonObjectFromAssistantText(assistantText: str) -> dict[str, Any]:
    # Model may wrap JSON in markdown fences; strip those before json.loads.
    cleanedText = assistantText.strip()
    if cleanedText.startswith("```"):
        lines = cleanedText.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleanedText = "\n".join(lines)
    return json.loads(cleanedText)


_PYREASON_BINARY_FACT_RE = re.compile(
    r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)\s*:\s*\[([^\]]+)\]",
    re.UNICODE,
)


def normalizePyreasonFactText(raw: str) -> str | None:
    # PyReason expects predicate(A,B):[lower,upper]. Remove stray braces or quotes after the closing bracket.
    s = (raw or "").strip()
    if not s:
        return None
    m = _PYREASON_BINARY_FACT_RE.match(s)
    if not m:
        return None
    pred, a, b, bound = m.group(1), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()
    boundNorm = ",".join(p.strip() for p in bound.split(","))
    return f"{pred}({a},{b}):[{boundNorm}]"


def sanitizePyreasonFactRows(rows: list[Any]) -> list[dict[str, Any]]:
    # Drop rows whose fact_text cannot be normalized (keeps PyReason add_fact_from_json from crashing).
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = str(row.get("fact_text", ""))
        ft = normalizePyreasonFactText(raw)
        if ft is None:
            continue
        cleaned.append({**row, "fact_text": ft})
    return cleaned


def readKnowledgeGraphContextLines(graphmlPath: Path) -> tuple[str, str]:
    # Human-readable node list and KG edge lines embedded in the LLM prompt (GraphML edge attrs with truth 1).
    knowledgeGraph = nx.read_graphml(graphmlPath)
    knowledgeGraph = nx.DiGraph(knowledgeGraph)
    edgeLines: list[str] = []
    for sourceNode, targetNode, attributeMap in knowledgeGraph.edges(data=True):
        for attributeKey, attributeValue in attributeMap.items():
            if attributeValue in (1, "1", 1.0):
                edgeLines.append(f"{attributeKey}({sourceNode},{targetNode})")
    sortedNodes = sorted(str(nodeId) for nodeId in knowledgeGraph.nodes())
    nodeListText = ", ".join(sortedNodes)
    edgeListText = "\n".join(sorted(edgeLines)) if edgeLines else "(no edges)"
    return nodeListText, edgeListText


def maxNaturalLanguageExamplesPerRuleFromEnvironment() -> int:
    # Caps how many English example lines we ask the model to produce per rule (1 to 5).
    rawValue = os.environ.get("LLM_NATURAL_LANGUAGE_EXAMPLES_PER_RULE", "5").strip()
    try:
        parsed = int(rawValue)
    except ValueError:
        return 5
    return max(1, min(5, parsed))


def maxPyreasonFactsPerRuleFromEnvironment() -> int:
    # Assignment: “up to five facts per rule” for grounding; default 5 (cap with env to save API cost).
    rawValue = os.environ.get("LLM_FACTS_PER_RULE", "5").strip()
    try:
        parsed = int(rawValue)
    except ValueError:
        return 5
    return max(1, min(5, parsed))


def supplementalNotesForRuleBody(ruleText: str) -> str:
    # Extra prompt text when rule body mentions citizenOf etc.
    separator = "<-1" if "<-1" in ruleText else "<-" if "<-" in ruleText else ""
    if not separator:
        return ""
    _head, body = ruleText.split(separator, 1)
    body = body.strip()
    notes: list[str] = []
    if "citizenOf(" in body:
        notes.append(
            "This rule’s body includes citizenOf: those edges are **inferred** by the citizenFromBirthCity rule from "
            "bornIn + cityIn — do **not** output citizenOf as an LLM fact. officialLanguage is already in the KG. "
            "You may use an empty pyreasonFacts array if the examples suffice, or only add bornIn / livesIn / worksAt when useful."
        )
    if not notes:
        return ""
    return "\nSPECIAL NOTES:\n- " + "\n- ".join(notes) + "\n"


def buildPromptForRule(
    ruleName: str,
    ruleText: str,
    ruleDescription: str,
    nodeListText: str,
    edgeListText: str,
    maxNaturalLanguageExamples: int,
    maxPyreasonFacts: int,
) -> str:
    # Full user message for one rule: KG context plus strict JSON schema instructions.
    extra = supplementalNotesForRuleBody(ruleText)
    return f"""You are helping with a university knowledge-graph assignment using PyReason.

PyReason edge facts look like: predicate(EntityA,EntityB):[1,1]
Use YAGO3-10 style identifiers: underscores, no spaces (e.g. Tom_Cruise, United_States).

RULE NAME: {ruleName}
LOGIC RULE (head inferred from body): {ruleText}
INTENT: {ruleDescription}
{extra}
KNOWN NODES IN THE GRAPH (you may ONLY use these as endpoints for NEW facts, unless you introduce a new person that connects to existing city/company nodes already listed):
{nodeListText}

EXISTING KG FACTS (do not duplicate the same predicate+pair; you may add missing body literals such as bornIn, livesIn, worksAt):
{edgeListText}

TASK:
- Propose up to {maxNaturalLanguageExamples} plausible REAL-WORLD style suggestions for this single rule (short English sentences in naturalLanguageExamples).
- Propose up to {maxPyreasonFacts} NEW PyReason edge fact(s) in pyreasonFacts using ONLY: bornIn, livesIn, worksAt (when they help ground **this** rule or the citizenship chain).
- Every subject/object in pyreasonFacts must appear in the KNOWN NODES list, except you may use a NEW person name as the FIRST argument of bornIn/livesIn/worksAt if the second argument is already a known node (use underscores).
- Each pyreasonFacts item’s fact_text must be **only** the fact string ending with ``:[1,1]`` — never paste extra characters after the closing ``]`` (no ``}}``, ``]}}``, or stray quotes from the JSON structure).

Return ONLY one valid JSON object (straight \" double quotes only, no trailing commas; pyreasonFacts may be []):
{{"ruleName":"{ruleName}","naturalLanguageExamples":["..."],"pyreasonFacts":[{{"fact_text":"bornIn(Some_Person,Some_City):[1,1]","name":"llmFactRuleUnique","static":true}}]}}
"""


def buildPayloadWithOpenAi(
    rulesFromDisk: list[dict[str, Any]],
    graphmlPath: Path,
    modelName: str,
    transcriptPath: Path | None = None,
) -> dict[str, Any]:
    # Loop rules, append facts and transcript entries, then sanitize all fact_text values before return.
    kgForCount = nx.DiGraph(nx.read_graphml(graphmlPath))
    kgNodeCount = kgForCount.number_of_nodes()
    kgEdgeCount = 0
    for _u, _v, attrs in kgForCount.edges(data=True):
        kgEdgeCount += sum(1 for val in attrs.values() if val in (1, "1", 1.0, True))
    print(
        f"  [llm] GraphML grounding: {kgNodeCount} node(s), {kgEdgeCount} KG edge(s) in prompts.",
        flush=True,
    )

    nodeListText, edgeListText = readKnowledgeGraphContextLines(graphmlPath)
    naturalLanguageByRuleName: dict[str, list[str]] = {}
    mergedPyreasonFacts: list[dict[str, Any]] = []
    transcriptEntries: list[dict[str, Any]] = []
    globalUsedFactNames: set[str] = set()
    maxNl = maxNaturalLanguageExamplesPerRuleFromEnvironment()
    maxFacts = maxPyreasonFactsPerRuleFromEnvironment()

    nRules = len(rulesFromDisk)
    for ruleIndex, ruleObject in enumerate(rulesFromDisk, start=1):
        ruleName = str(ruleObject.get("name") or "unnamedRule")
        ruleText = str(ruleObject["rule_text"])
        ruleDescription = str(ruleObject.get("description") or "")
        print(f"  [llm] ({ruleIndex}/{nRules}) Requesting facts for rule “{ruleName}”…", flush=True)
        userPromptText = buildPromptForRule(
            ruleName,
            ruleText,
            ruleDescription,
            nodeListText,
            edgeListText,
            maxNl,
            maxFacts,
        )
        parsedObject: dict[str, Any] | None = None
        assistantText = ""
        lastParseExc: json.JSONDecodeError | None = None
        for apiTry in range(3):
            responseJson, openAiErrorDetail = tryFetchOpenAiChatJson(userPromptText, modelName=modelName)
            if responseJson is None:
                raise RuntimeError(openAiErrorDetail or "OpenAI request failed.")
            assistantText = extractAssistantTextFromChatCompletion(responseJson) or ""
            if not assistantText:
                raise RuntimeError("OpenAI response had no assistant text.")
            try:
                parsedObject = parseJsonObjectFromAssistantText(assistantText)
                break
            except json.JSONDecodeError as exc:
                lastParseExc = exc
                print(
                    f"  [llm] ({ruleIndex}/{nRules}) JSON parse failed for “{ruleName}”, "
                    f"retry {apiTry + 1}/3…",
                    flush=True,
                )
        if parsedObject is None:
            raise RuntimeError(
                f"Model returned invalid JSON for rule {ruleName} after 3 tries: {lastParseExc!s}. "
                f"First 500 chars: {assistantText[:500]!r}"
            ) from lastParseExc
        examples = parsedObject.get("naturalLanguageExamples") or []
        if isinstance(examples, list):
            naturalLanguageByRuleName[ruleName] = [str(item) for item in examples[:maxNl]]
        factRows = parsedObject.get("pyreasonFacts") or []
        if not isinstance(factRows, list):
            factRows = []
        factCountBefore = len(mergedPyreasonFacts)
        for index, factRow in enumerate(factRows[:maxFacts]):
            if not isinstance(factRow, dict):
                continue
            factName = str(factRow.get("name") or f"llmFact_{ruleName}_{index}")
            originalName = factName
            suffix = 0
            while factName in globalUsedFactNames:
                suffix += 1
                factName = f"{originalName}_{suffix}"
            globalUsedFactNames.add(factName)
            mergedPyreasonFacts.append(
                {
                    "fact_text": str(factRow["fact_text"]),
                    "name": factName,
                    "static": bool(factRow.get("static", True)),
                }
            )
        nAdded = len(mergedPyreasonFacts) - factCountBefore
        print(
            f"  [llm] ({ruleIndex}/{nRules}) OK “{ruleName}” — added {nAdded} fact row(s) "
            f"({len(mergedPyreasonFacts)} cumulative).",
            flush=True,
        )
        transcriptEntries.append(
            {
                "ruleName": ruleName,
                "openAiUserPrompt": userPromptText,
                "openAiAssistantReply": assistantText,
                "parsedJson": parsedObject,
            }
        )

    beforeSanitize = len(mergedPyreasonFacts)
    mergedPyreasonFacts = sanitizePyreasonFactRows(mergedPyreasonFacts)
    dropped = beforeSanitize - len(mergedPyreasonFacts)
    if dropped:
        print(
            f"  [llm] Dropped {dropped} fact row(s) with unparseable fact_text after sanitize.",
            flush=True,
        )

    targetTranscriptPath = transcriptPath or paths.openAiTranscriptPath(write=True)
    targetTranscriptPath.parent.mkdir(parents=True, exist_ok=True)
    with targetTranscriptPath.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {
                    "model": modelName,
                    "maxNaturalLanguageExamplesPerRule": maxNl,
                    "maxPyreasonFactsPerRule": maxFacts,
                },
                "entries": transcriptEntries,
            },
            handle,
            indent=2,
        )

    return {
        "metadata": {
            "generatedBy": "openai_chat_completions",
            "model": modelName,
            "maxNaturalLanguageExamplesPerRule": maxNl,
            "maxPyreasonFactsPerRule": maxFacts,
        },
        "naturalLanguageByRuleName": naturalLanguageByRuleName,
        "pyreasonFacts": mergedPyreasonFacts,
    }


def resolveLlmPayload(
    graphmlPath: Path | None = None,
    rulesPath: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Tuple is (mergedPayloadDict, smallMetadataDict for inference_result llmGeneration block).
    loadRepositoryEnvironment()
    graphPath = graphmlPath or paths.graphmlPath()
    rulesFilePath = rulesPath or paths.rulesPath()
    envFilePath = paths.repositoryEnvFilePath()

    apiKey = readOpenAiApiKeyFromEnvironment()
    print("[llm] Validating API key and GraphML path…", flush=True)
    if not apiKey:
        raise RuntimeError(
            f"OPENAI_API_KEY is required. Add it to {envFilePath} (see .env.example). "
            "Offline LLM mode was removed from this project."
        )
    if not graphPath.is_file():
        raise RuntimeError(
            f"GraphML missing at {graphPath}. Run: python -m src.buildYagoSubsetFromHuggingFace (or restore yago_subset.graphml)."
        )

    modelName = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    with rulesFilePath.open(encoding="utf-8") as handle:
        rulesFromDisk = paths.filterShowableRules(json.load(handle))
    if not rulesFromDisk:
        raise RuntimeError(
            f"No showable rules in {rulesFilePath.name}. Set \"showable\": true on at least one rule."
        )
    print(
        f"[llm] Calling OpenAI Chat Completions ({modelName}) — "
        f"{len(rulesFromDisk)} rule(s), one HTTP request per rule (network latency is normal).",
        flush=True,
    )
    transcriptPath = paths.openAiTranscriptPath(write=True)
    payload = buildPayloadWithOpenAi(
        rulesFromDisk,
        graphPath,
        modelName=modelName,
        transcriptPath=transcriptPath,
    )
    nFacts = len(payload.get("pyreasonFacts") or [])
    print(
        f"[llm] OpenAI phase done. Merged {nFacts} PyReason fact row(s); transcript: {transcriptPath.name}",
        flush=True,
    )
    return payload, {
        "mode": "openai",
        "model": modelName,
        "openAiTranscriptRelativePath": str(transcriptPath.relative_to(paths.repositoryRootDirectory)),
    }


def writePyreasonFactFileFromPayload(llmPayload: dict[str, Any], destinationPath: Path) -> Path:
    # Writes only the fact rows array; also replaces pyreasonFacts inside llmPayload with sanitized copies.
    factsList = llmPayload.get("pyreasonFacts")
    if factsList is None:
        raise ValueError("LLM payload is missing 'pyreasonFacts' array.")
    cleaned = sanitizePyreasonFactRows(factsList if isinstance(factsList, list) else [])
    llmPayload["pyreasonFacts"] = cleaned
    destinationPath.parent.mkdir(parents=True, exist_ok=True)
    with destinationPath.open("w", encoding="utf-8") as handle:
        json.dump(cleaned, handle, indent=2)
    return destinationPath
