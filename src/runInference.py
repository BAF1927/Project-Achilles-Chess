# End-to-end assessment run: refresh GraphML if needed, call OpenAI for facts per rule, run PyReason, write outputs.
# Steps in runFullAssessment: ensure graph file exists; resolveLlmPayload builds facts and transcript;
# PyReason loads graphml, temporary rules JSON, fact JSON, then reason(); export trace CSV and inference_result.json.
# All writes use artifactDirectory with write=True (src/output). Dashboard demo mode only changes its read folder.
from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Any
import pyreason as pr
from . import buildYagoSubsetFromHuggingFace, llmClient, paths, rule_display

repositoryRootDirectory = paths.repositoryRootDirectory

def pathRelativeToRepository(absolutePath: Path) -> str:
    # Store paths in inference_result.json relative to repo root for portability.
    try:
        return str(absolutePath.relative_to(repositoryRootDirectory))
    except ValueError:
        return str(absolutePath)

# Rule heads we export from the last PyReason timestep (see collectInferredEdges).
headPredicateLabels = [
    "citizenOf",
    "livesInCountry",
    "employerCountry",
    "speaksLanguage",
]

def ensureGraphmlExists() -> Path:
    # HF YAGO3-10 slice when GraphML missing; never replaces existing file.
    graphPath = paths.graphmlPath()
    if graphPath.is_file():
        print(f"[graph] Using existing GraphML: {graphPath}", flush=True)
        return graphPath
    print(
        "[graph] No GraphML on disk — streaming YAGO3-10 from Hugging Face (can take a minute)…",
        flush=True,
    )
    try:
        buildYagoSubsetFromHuggingFace.writeGraphmlFromHuggingFace(graphPath)
        print(f"[graph] Wrote {graphPath}", flush=True)
        return graphPath
    except Exception as exc:
        raise RuntimeError(
            f"Could not build assignment GraphML from Hugging Face ({exc!s}). "
            "Install `datasets`, check the network, then run: python -m src.buildYagoSubsetFromHuggingFace"
        ) from exc

def stripRulesForPyreason(rulesWithNotes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # PyReason loader only needs rule_text, name, infer_edges, set_static (drops UI-only fields).
    strippedRules: list[dict[str, Any]] = []
    for ruleObject in rulesWithNotes:
        strippedRules.append(
            {
                "rule_text": ruleObject["rule_text"],
                "name": ruleObject.get("name"),
                "infer_edges": ruleObject.get("infer_edges", True),
                "set_static": ruleObject.get("set_static", False),
            }
        )
    return strippedRules

def writeTemporaryRulesJson(rulesData: list[dict[str, Any]], temporaryPath: Path) -> None:
    # add_rule_from_json reads this file from disk inside src/output.
    temporaryPath.parent.mkdir(parents=True, exist_ok=True)
    with temporaryPath.open("w", encoding="utf-8") as handle:
        json.dump(rulesData, handle, indent=2)


def ruleTraceEdgeToRecords(ruleTraceEdge: Any) -> list[dict[str, Any]]:
    # Turn PyReason trace tuples into plain dicts for JSON (inference_result sample rows).
    records: list[dict[str, Any]] = []
    for row in ruleTraceEdge:
        edgePair = row[2]
        labelValue = row[3]
        boundValue = row[4]
        records.append(
            {
                "timeStep": int(row[0]),
                "fixedPointOperation": int(row[1]),
                "sourceNode": str(edgePair[0]),
                "targetNode": str(edgePair[1]),
                "label": str(labelValue),
                "boundLower": float(boundValue.lower),
                "boundUpper": float(boundValue.upper),
                "occurredDueTo": str(row[6]) if len(row) > 6 else "",
            }
        )
    return records


def collectInferredEdges(interpretation: Any) -> dict[str, list[dict[str, Any]]]:
    # For each head predicate, take the latest timestep frame that still has edges (PyReason API).
    inferredByPredicate: dict[str, list[dict[str, Any]]] = {}
    for predicateLabel in headPredicateLabels:
        framesPerTimestep = pr.filter_and_sort_edges(interpretation, [predicateLabel])
        chosenFrame = None
        for frame in reversed(framesPerTimestep):
            if frame is not None and not frame.empty:
                chosenFrame = frame
                break
        rows: list[dict[str, Any]] = []
        if chosenFrame is not None:
            for item in chosenFrame.to_dict("records"):
                component = item.get("component")
                if component is None:
                    continue
                boundInterval = item.get(predicateLabel)
                rows.append(
                    {
                        "sourceNode": str(component[0]),
                        "targetNode": str(component[1]),
                        "bound": str(boundInterval),
                    }
                )
        inferredByPredicate[predicateLabel] = rows
    return inferredByPredicate


def runFullAssessment() -> dict[str, Any]:
    # Single entry: LLM phase, then PyReason, then write CSV trace and inference_result.json bundle.
    print("=== runInference: start ===", flush=True)
    print("[env] Loading .env from repo root (if present)…", flush=True)
    llmClient.loadRepositoryEnvironment()
    print("[env] Done.", flush=True)

    ensureGraphmlExists()

    print(
        "\n--- OpenAI: generating grounding facts (this block is often 30s–3min; waits are usually the API) ---\n",
        flush=True,
    )
    llmPayload, llmGenerationMetadata = llmClient.resolveLlmPayload(
        graphmlPath=paths.graphmlPath(),
        rulesPath=paths.rulesPath(),
    )
    outWrite = paths.artifactDirectory(write=True)
    outWrite.mkdir(parents=True, exist_ok=True)
    factDumpPath = outWrite / "llm_facts_pyreason.json"
    print(f"[files] Writing LLM facts → {factDumpPath.name} …", flush=True)
    llmClient.writePyreasonFactFileFromPayload(llmPayload, factDumpPath)
    nPayloadFacts = len(llmPayload.get("pyreasonFacts") or [])
    print(f"[files] Wrote {nPayloadFacts} sanitized fact row(s).", flush=True)
    fullLlmPayloadPath = outWrite / "llm_full_payload.json"
    print(f"[files] Writing merged LLM payload → {fullLlmPayloadPath.name} …", flush=True)
    with fullLlmPayloadPath.open("w", encoding="utf-8") as handle:
        json.dump(llmPayload, handle, indent=2)

    rulesFile = paths.rulesPath()
    with rulesFile.open(encoding="utf-8") as handle:
        rulesFromDisk = paths.filterShowableRules(json.load(handle))
    if not rulesFromDisk:
        raise RuntimeError(
            f"No showable rules in {rulesFile.name}. Set \"showable\": true (or omit) on rules you want to run."
        )
    print(f"[rules] Loaded {len(rulesFromDisk)} showable rule(s) from {rulesFile.name}.", flush=True)
    temporaryRulesPath = outWrite / "rules_for_pyreason_loader.json"
    writeTemporaryRulesJson(stripRulesForPyreason(rulesFromDisk), temporaryRulesPath)
    print(f"[rules] Wrote PyReason loader copy → {temporaryRulesPath.name}", flush=True)

    print("\n--- PyReason: loading graph, rules, facts (in-memory; starts clean from pr.reset) ---\n", flush=True)
    pr.reset()
    pr.reset_settings()
    pr.settings.verbose = False
    pr.settings.store_interpretation_changes = True
    pr.settings.save_graph_attributes_to_trace = False
    print("[pyreason] Interpreter reset.", flush=True)

    print("[pyreason] load_graphml …", flush=True)
    pr.load_graphml(str(paths.graphmlPath()))
    print("[pyreason] load_graphml returned.", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("[pyreason] add_rule_from_json …", flush=True)
        pr.add_rule_from_json(str(temporaryRulesPath), raise_errors=True)
    print(f"[pyreason] Rules loaded ({len(rulesFromDisk)}).", flush=True)
    print(f"[pyreason] add_fact_from_json ({nPayloadFacts} LLM facts) …", flush=True)
    pr.add_fact_from_json(str(factDumpPath), raise_errors=True)
    print("[pyreason] Facts loaded.", flush=True)

    print(
        "\n[pyreason] reason(timesteps=6) starting — **this is often the longest step**. "
        "First time after install, Numba may compile for 1–3+ minutes with little output; later runs are faster.\n",
        flush=True,
    )
    interpretation = pr.reason(
        timesteps=6,
        convergence_threshold=-1,
        convergence_bound_threshold=-1,
        queries=None,
        again=False,
        restart=True,
    )
    print("[pyreason] reason() finished.", flush=True)

    print("[pyreason] get_rule_trace …", flush=True)
    _nodeTraceDataframe, edgeTraceDataframe = pr.get_rule_trace(interpretation)
    nTrace = len(interpretation.rule_trace_edge)

    textualTracePath = outWrite / "inference_rule_trace_edges.csv"
    outWrite.mkdir(parents=True, exist_ok=True)
    print(f"[files] Writing trace CSV ({nTrace} rows) → {textualTracePath.name} …", flush=True)
    with textualTracePath.open("w", encoding="utf-8") as handle:
        handle.write(edgeTraceDataframe.to_csv(index=False))

    print("[files] Building result bundle (inferred head edges + trace sample) …", flush=True)
    inferredEdges = collectInferredEdges(interpretation)
    inferredWatchable = rule_display.inferredEdgesWatchableList(
        inferredEdges,
        rulesFromDisk,
        predicateOrder=headPredicateLabels,
    )
    resultBundle: dict[str, Any] = {
        "graphmlRelativePath": pathRelativeToRepository(paths.graphmlPath()),
        "rulesRelativePath": pathRelativeToRepository(rulesFile),
        "llmGeneration": llmGenerationMetadata,
        "llmFullPayloadRelativePath": pathRelativeToRepository(fullLlmPayloadPath),
        "llmFactsWrittenRelativePath": pathRelativeToRepository(factDumpPath),
        "ruleTraceEdgesCsvRelativePath": pathRelativeToRepository(textualTracePath),
        "inferredEdgesByPredicate": inferredEdges,
        "inferredEdgesWatchable": inferredWatchable,
        "ruleTraceEdgeRecordCount": len(interpretation.rule_trace_edge),
        "ruleTraceEdgeSample": ruleTraceEdgeToRecords(interpretation.rule_trace_edge[:40]),
    }

    resultPath = paths.inferenceResultPath(write=True)
    print(f"[files] Writing {resultPath.name} …", flush=True)
    with resultPath.open("w", encoding="utf-8") as handle:
        json.dump(resultBundle, handle, indent=2)

    print("=== runInference: finished successfully ===", flush=True)
    return resultBundle


if __name__ == "__main__":
    summary = runFullAssessment()
    print(json.dumps(summary, indent=2))
