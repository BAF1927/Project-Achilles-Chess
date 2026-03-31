# Streamlit web UI for the assessment: loads saved files only (no PyReason, no OpenAI calls).
# Graph tab builds a PyVis HTML chart from YAGO GraphML plus LLM facts JSON plus inferred edges JSON.
# Paths come from paths.artifactDirectory(write=False): normally src/output, or src/brunoOutput for frozen demos.
# Run from repo root: streamlit run src/dashboard.py
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Streamlit may start with cwd somewhere else; ensure imports like "from src import paths" resolve.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import networkx as nx
import pandas as pd
import streamlit as st
from pyvis.network import Network
from src import llmClient, paths, rule_display

# Trace / inference table columns hidden in the UI (noisy or redundant).
TRACE_COLUMNS_DROP = ("Old Bound", "Occurred Due To", "Inconsistency Message")
INFER_SAMPLE_DROP = ("occurredDueTo",)

# Parse first binary fact in each LLM row: predicate(A, B)
FACT_HEAD_RE = re.compile(
    r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)\s*",
    re.UNICODE,
)

# PyVis edge stroke colors: solid lines for YAGO GraphML, dashed for LLM json facts (matches sidebar tables).
PREDICATE_COLORS_KG = {
    "cityIn": "#4ecdc4",
    "officialLanguage": "#ffe66d",
    "companyHeadquarteredIn": "#ff6b6b",
}
PREDICATE_COLORS_LLM = {
    "citizenOf": "#f4a261",
    "bornIn": "#a8e6cf",
    "livesIn": "#5096b8",
    "worksAt": "#a855d7",
}

# Inferred rule heads: brown default; speaksLanguage stands out in white.
INFERRED_HEAD_EDGE_COLORS = {
    "speaksLanguage": "#ffffff",
}
INFERRED_HEAD_EDGE_COLOR_DEFAULT = "#a0522d"  # sienna; distinct from #ffe66d / #eab308

# Node fill by inferred entity kind (infer_node_entity_kinds).
NODE_ENTITY_COLORS = {
    "person": "#22c55e",
    "city": "#3b82f6",
    "country": "#eab308",
    "company": "#ef4444",
    "language": "#c084fc",
    "unknown": "#64748b",
}
NODE_ENTITY_LABEL = {
    "person": "Person",
    "city": "City",
    "country": "Country / region",
    "company": "Workplace (company)",
    "language": "Language",
    "unknown": "Unknown",
}


def loadJson(jsonPath: Path) -> object | None:
    # None if file missing; otherwise parsed JSON (object, list, or scalar).
    if not jsonPath.is_file():
        return None
    with jsonPath.open(encoding="utf-8") as f:
        return json.load(f)

def inferNodeEntityKinds(combined: nx.MultiDiGraph) -> dict[str, str]:
    # Node id to coarse type: person, city, country, company, language, or unknown (vote from incident edges).
    from collections import defaultdict

    scores: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for u, v, _k, data in combined.edges(keys=True, data=True):
        pred = str(data.get("predicate", ""))
        kind = data.get("edge_kind", "kg")
        if kind == "kg":
            if pred == "cityIn":
                scores[str(u)]["city"] += 2
                scores[str(v)]["country"] += 2
            elif pred == "officialLanguage":
                scores[str(u)]["country"] += 2
                scores[str(v)]["language"] += 2
            elif pred == "companyHeadquarteredIn":
                scores[str(u)]["company"] += 2
                scores[str(v)]["country"] += 2
        elif kind == "llm":
            if pred == "bornIn":
                scores[str(u)]["person"] += 2
                scores[str(v)]["city"] += 2
            elif pred == "livesIn":
                scores[str(u)]["person"] += 2
                scores[str(v)]["city"] += 2
            elif pred == "worksAt":
                scores[str(u)]["person"] += 2
                scores[str(v)]["company"] += 2
            elif pred == "citizenOf":
                scores[str(u)]["person"] += 2
                scores[str(v)]["country"] += 2
        elif kind == "inferred":
            if pred == "citizenOf":
                scores[str(u)]["person"] += 1
                scores[str(v)]["country"] += 1
            elif pred in ("livesInCountry", "employerCountry"):
                scores[str(u)]["person"] += 1
                scores[str(v)]["country"] += 1
            elif pred == "speaksLanguage":
                scores[str(u)]["person"] += 1
                scores[str(v)]["language"] += 1

    tieBreakOrder = ("person", "company", "city", "country", "language")
    out: dict[str, str] = {}
    for nid in combined.nodes():
        nodeIdStr = str(nid)
        bucket = scores.get(nodeIdStr, {})
        if not bucket:
            out[nodeIdStr] = "unknown"
            continue
        bestScore = max(bucket.values())
        candidates = [r for r, c in bucket.items() if c == bestScore]
        if len(candidates) == 1:
            out[nodeIdStr] = candidates[0]
        else:
            picked = "unknown"
            for role in tieBreakOrder:
                if role in candidates:
                    picked = role
                    break
            out[nodeIdStr] = picked
    return out


def dropColumnsIfPresent(dataFrame: pd.DataFrame, columnNames: tuple[str, ...]) -> pd.DataFrame:
    # No error if a column name is absent; returns a copy unchanged when nothing to drop.
    toDrop = [c for c in columnNames if c in dataFrame.columns]
    return dataFrame.drop(columns=toDrop) if toDrop else dataFrame


def showableRuleNameSet(showable: list[dict[str, Any]]) -> set[str]:
    # Internal rule "name" ids from rules.json for filtering LLM payload and transcript rows.
    return {str(r["name"]) for r in showable if r.get("name")}


def addRuleTitleColumn(
    dataFrame: pd.DataFrame,
    showableRules: list[dict[str, Any]],
    *,
    labelColumn: str,
    triggeredColumn: str,
) -> pd.DataFrame:
    # Adds "From Rule" column: display_title when PyReason says Rule; literal label when it says Fact.
    predTitles = rule_display.predicateToWatchableTitles(showableRules)
    frame = dataFrame.copy()

    def cellForRow(row: pd.Series) -> str:
        trig = str(row.get(triggeredColumn, "")).strip()
        lab = str(row.get(labelColumn, "")).strip()
        trigLower = trig.lower()
        if trigLower == "fact":
            return "Previously Known Fact"
        if trigLower == "rule":
            return predTitles.get(lab, lab or "—")
        if not trig or trig == "-":
            return "—"
        return f"{trig} · {predTitles.get(lab, lab)}"

    titles = frame.apply(cellForRow, axis=1)
    frame.insert(0, "From Rule", titles)
    return frame


def mergePyreasonInferredIntoCombined(
    combined: nx.MultiDiGraph,
    inferencePath: Path | None,
) -> None:
    # Mutates combined graph in place with edge_kind "inferred" and rule titles for tooltips.
    if inferencePath is None or not inferencePath.is_file():
        return
    payload = loadJson(inferencePath)
    if not isinstance(payload, dict):
        return
    blocks = payload.get("inferredEdgesWatchable")
    if not isinstance(blocks, list):
        blocks = []
        raw = payload.get("inferredEdgesByPredicate")
        if isinstance(raw, dict):
            for predKey, rows in raw.items():
                if isinstance(rows, list):
                    blocks.append(
                        {
                            "headPredicate": predKey,
                            "watchableRuleTitles": [],
                            "edges": rows,
                        }
                    )
    for block in blocks:
        pred = str(block.get("headPredicate") or "")
        titles = block.get("watchableRuleTitles") or []
        watchable = " · ".join(str(t) for t in titles) if titles else pred
        for row in block.get("edges") or []:
            if not isinstance(row, dict):
                continue
            a, b = row.get("sourceNode"), row.get("targetNode")
            if a is None or b is None:
                continue
            aStr, bStr = str(a), str(b)
            combined.add_node(aStr)
            combined.add_node(bStr)
            combined.add_edge(
                aStr,
                bStr,
                predicate=pred,
                edge_kind="inferred",
                watchable_label=watchable,
            )


def combinedLayoutGraph(graphmlPath: Path, llmFactsPath: Path | None) -> nx.MultiDiGraph:
    # One MultiDiGraph carrying YAGO edges (edge_kind kg) and parsed LLM facts (edge_kind llm).
    base = nx.DiGraph(nx.read_graphml(graphmlPath))
    combined = nx.MultiDiGraph()
    for n in base.nodes():
        combined.add_node(n)
    for u, v, data in base.edges(data=True):
        for key, val in data.items():
            if val in (1, "1", 1.0, True):
                pred = str(key)
                combined.add_edge(u, v, predicate=pred, edge_kind="kg")

    if llmFactsPath and llmFactsPath.is_file():
        raw = loadJson(llmFactsPath)
        if isinstance(raw, list):
            for row in raw:
                if not isinstance(row, dict):
                    continue
                ft = str(row.get("fact_text", ""))
                m = FACT_HEAD_RE.match(ft)
                if not m:
                    continue
                pred, a, b = m.group(1), m.group(2).strip(), m.group(3).strip()
                combined.add_node(a)
                combined.add_node(b)
                combined.add_edge(a, b, predicate=pred, edge_kind="llm")

    return combined


def graphmlToPyvisHtml(
    graphmlPath: Path,
    llmFactsPath: Path | None,
    inferencePath: Path | None = None,
) -> str:
    # Full HTML string for st.components.v1.html: fixed spring positions, physics disabled in PyVis.
    baseKg = nx.DiGraph(nx.read_graphml(graphmlPath))
    kgNodeSet = set(baseKg.nodes())
    combined = combinedLayoutGraph(graphmlPath, llmFactsPath)
    mergePyreasonInferredIntoCombined(combined, inferencePath)

    nodeCount = max(combined.number_of_nodes(), 1)
    spreadK = max(2.8, 9.0 / (nodeCount**0.5))
    pos = nx.spring_layout(combined, k=spreadK, iterations=400, seed=42, scale=1.0)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minX, maxX = min(xs), max(xs)
    minY, maxY = min(ys), max(ys)
    spanX = max(maxX - minX, 0.2)
    spanY = max(maxY - minY, 0.2)
    widthPx, heightPx = 1000, 680
    margin = 90

    def toPx(xy: tuple[float, float]) -> tuple[float, float]:
        x, y = xy
        px = margin + (x - minX) / spanX * (widthPx - 2 * margin)
        py = margin + (y - minY) / spanY * (heightPx - 2 * margin)
        return float(px), float(py)

    net = Network(
        height=f"{heightPx}px",
        width="100%",
        directed=True,
        bgcolor="#16161e",
        font_color="#e0e0e0",
    )

    entityKind = inferNodeEntityKinds(combined)
    for nid in combined.nodes():
        inKg = nid in kgNodeSet
        px, py = toPx(pos[nid])
        kind = entityKind.get(str(nid), "unknown")
        kindLabel = NODE_ENTITY_LABEL.get(kind, kind)
        net.add_node(
            nid,
            label=nid.replace("_", " "),
            title=(
                f"{kindLabel}: {nid}\n"
                f"({'also in YAGO GraphML' if inKg else 'not in GraphML alone — appears via facts / inference'})"
            ),
            x=px,
            y=py,
            color=NODE_ENTITY_COLORS.get(kind, NODE_ENTITY_COLORS["unknown"]),
            size=22,
        )

    for u, v, _k, data in combined.edges(keys=True, data=True):
        pred = str(data.get("predicate", ""))
        kind = data.get("edge_kind", "kg")
        if kind == "inferred":
            watch = str(data.get("watchable_label", ""))
            tip = f"INFERRED {pred} — {watch} (PyReason rule head; assignment deliverable)"
            edgeHex = INFERRED_HEAD_EDGE_COLORS.get(pred, INFERRED_HEAD_EDGE_COLOR_DEFAULT)
            net.add_edge(
                u,
                v,
                title=tip,
                color={"color": edgeHex, "highlight": "#22c55e"},
                width=4,
                dashes=False,
            )
            continue
        isLlm = kind == "llm"
        palette = PREDICATE_COLORS_LLM if isLlm else PREDICATE_COLORS_KG
        color = palette.get(pred, "#94a3b8")
        tip = f"{pred} — {'LLM grounding fact' if isLlm else 'YAGO subset (GraphML)'}"
        net.add_edge(
            u,
            v,
            title=tip,
            color={"color": color, "highlight": "#ffffff"},
            width=2 if not isLlm else 2,
            dashes=isLlm,
        )

    # PyVis JSON options: allow drag and zoom; edge labels off by default (use hover title).
    net.set_options(
        """
{
  "physics": { "enabled": false },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "zoomView": true,
    "dragView": true,
    "dragNodes": true,
    "selectConnectedEdges": false
  },
  "edges": {
    "smooth": { "type": "dynamic", "roundness": 0.5 },
    "arrows": { "to": { "enabled": true, "scaleFactor": 0.65 } },
    "font": { "size": 0 }
  },
  "nodes": { "font": { "size": 13, "face": "system-ui" }, "borderWidth": 1, "borderWidthSelected": 2 }
}
"""
    )

    return net.generate_html()


def main() -> None:
    # Five tabs below; each reads only from disk paths resolved after this block.
    llmClient.loadRepositoryEnvironment()
    st.set_page_config(
        page_title="Assessment — Visual Overview",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Assessment — Visual Overview")
    st.caption(
        "Leibniz Lab — read-only dashboard. Drag nodes to untangle, hover arrows for predicates, scroll to zoom."
    )

    graphPath = paths.graphmlPath()
    rulesPath = paths.rulesPath()
    rulesRaw = loadJson(rulesPath)
    showableList = paths.filterShowableRules(rulesRaw or [])
    showableNames = showableRuleNameSet(showableList)

    artifactRoot = paths.artifactDirectory(write=False)
    factsPath = artifactRoot / "llm_facts_pyreason.json"
    fullPayloadPath = artifactRoot / "llm_full_payload.json"
    inferencePath = artifactRoot / paths.inferenceResultFileName
    traceCsv = artifactRoot / "inference_rule_trace_edges.csv"
    transcriptPath = artifactRoot / paths.openAiTranscriptFileName

    # Sidebar: explain demo mode, list artifact paths, long legend for graph colors (user-facing markdown).
    with st.sidebar:
        st.header("What This App Does")
        if os.environ.get("LEIBNIZ_USE_BRUNO_OUTPUT", "").strip().lower() in ("1", "true", "yes"):
            st.success(
                "Demo mode: showing frozen files under `src/brunoOutput/`. "
                "Unset `LEIBNIZ_USE_BRUNO_OUTPUT` in `.env` if you want the live `src/output/` run instead."
            )
        st.info(
            "Everything you see is loaded from files on disk (JSON, GraphML, CSV). "
            "**It does not run PyReason, call OpenAI, or change files.** "
            "After `python -m src.runInference`, reload the page in your browser to pick up new outputs."
        )
        st.header("Files")
        st.write(f"**GraphML:** `{graphPath.relative_to(_ROOT)}`")
        st.write(f"**Rules:** `{rulesPath.relative_to(_ROOT)}`")
        st.write(f"**Facts:** `{factsPath.relative_to(_ROOT)}`")
        st.write(f"**Inference:** `{inferencePath.relative_to(_ROOT)}`")

        st.header("Knowledge Graph Tab — What You Are Looking At")
        st.markdown(
            """
Three layers in one view:

1. **yago_subset.graphml** — YAGO slice (KG in PyReason)  
2. **llm_facts_pyreason.json** — LLM facts (**bornIn**, **livesIn**, **worksAt**, ...)  
3. **inference_result.json** — thick brown / white edges are **inferred rule heads**

**Node colors** (entity kind):

| Kind | Color |
|------|--------|
| **Person** | Green (#22c55e) |
| **City** | Blue (#3b82f6) |
| **Country / region** | Yellow (#eab308) |
| **Workplace (company)** | Red (#ef4444) |
| **Language** | Purple (#c084fc) |
| **Unknown** | Gray (#64748b) |

Hover a node for where it came from (GraphML vs facts).

**Edges:** direction is source to target (arrowhead). Solid = from GraphML. Dashed = from the LLM fact JSON. Hover an edge for the predicate name.

*GraphML (solid):*
| Predicate | Meaning | Color |
|-----------|---------|-------|
| **cityIn** | City belongs to a country or region | #4ecdc4 |
| **officialLanguage** | Country has that language | #ffe66d |
| **companyHeadquarteredIn** | Company based in that country | #ff6b6b |

*LLM file (dashed):*
| Predicate | Meaning | Color |
|-----------|---------|-------|
| **bornIn** | Person born in that city | #a8e6cf |
| **livesIn** | Person lives in that city | #5096b8 |
| **worksAt** | Person works at that company | #a855d7 |
| **citizenOf** | Person citizen of that country (if present in LLM JSON) | #f4a261 |

*After PyReason (**reason()**), inferred heads from **inference_result.json** (thick):*
| Predicate | Color |
|-----------|--------|
| **citizenOf**, **livesInCountry**, **employerCountry**, ... | Brown (#a0522d) |
| **speaksLanguage** | White (#ffffff) |

**Trace Samples** tab: step log from PyReason (one row per edge update).
"""
        )

    tabGraph, tabRules, tabLlm, tabInfer, tabTrace = st.tabs(
        ["Knowledge Graph", "Rules", "LLM Facts", "Inference", "Trace Samples"]
    )

    with tabGraph:
        # Interactive PyVis network (HTML embed).
        if not graphPath.is_file():
            st.error("GraphML not found. Run `python -m src.buildYagoSubsetFromHuggingFace` or inference once.")
        else:
            st.caption(
                "Drag nodes to move them, pan and zoom on the background, hover or click edges to see what they represent."
            )
            graphHtml = graphmlToPyvisHtml(graphPath, factsPath, inferencePath)
            st.components.v1.html(graphHtml, height=720, scrolling=True)

    with tabRules:
        # rules.json entries where showable is not false: title, description, rule_text code blocks.
        if not showableList:
            st.warning("No **showable** rules in `rules.json` (set `\"showable\": true` or omit the field).")
        else:
            nShow = len(showableList)
            st.subheader("Rules")
            if nShow != 5:
                st.warning(
                    "The lab brief asks for **five** rules. Adjust `src/data/rules.json` "
                    "or mark extra rules as `\"showable\": false` if they should not appear in this UI."
                )
            for ruleIndex, ruleRow in enumerate(showableList):
                title = rule_display.display_title_for_rule(ruleRow)
                st.markdown(f"**{title}**")
                if ruleRow.get("description"):
                    st.write(ruleRow["description"])
                st.caption("PyReason Syntax")
                st.code(ruleRow.get("rule_text", "") or "", language="text")
                if ruleIndex < nShow - 1:
                    st.divider()

    with tabLlm:
        # Tables for llm_facts_pyreason.json, llm_full_payload.json, and optional OpenAI transcript JSON.
        payload = loadJson(fullPayloadPath)
        facts = loadJson(factsPath)
        transcript = loadJson(transcriptPath)

        if facts is None and payload is None:
            st.warning("No LLM output files yet. Run `python -m src.runInference`.")
        if isinstance(facts, list) and facts:
            st.subheader("PyReason Fact File (`llm_facts_pyreason.json`)")
            st.dataframe(pd.DataFrame(facts), use_container_width=True, hide_index=True)

        if isinstance(payload, dict):
            st.subheader("Natural Language By Rule (`llm_full_payload.json`)")
            nl = payload.get("naturalLanguageByRuleName") or payload.get("natural_language_by_rule")
            if isinstance(nl, dict):
                titleByName = {
                    str(r.get("name")): rule_display.displayTitleForRule(r) for r in showableList
                }
                for ruleRow in showableList:
                    ruleName = ruleRow.get("name")
                    if not ruleName or ruleName not in nl:
                        continue
                    examples = nl[ruleName]
                    st.markdown(f"**{titleByName.get(str(ruleName), ruleName)}**")
                    if isinstance(examples, list):
                        for ex in examples:
                            st.write(f"- {ex}")
                    else:
                        st.write(examples)
                staleKeys = [k for k in nl if k not in showableNames]
                if staleKeys:
                    st.caption(
                        f"Hiding LLM examples for non-showable or removed rule ids: {', '.join(sorted(staleKeys))}"
                    )
            meta = payload.get("metadata")
            if meta:
                st.json(meta)

        if isinstance(transcript, dict) and transcript.get("entries"):
            st.subheader("OpenAI Transcript")
            entriesAll = transcript["entries"]
            entries = [e for e in entriesAll if isinstance(e, dict) and e.get("ruleName") in showableNames]
            titleByName = {
                str(r.get("name")): rule_display.displayTitleForRule(r) for r in showableList
            }
            if not entries:
                st.info(
                    "No transcript entries match **showable** rules (refresh outputs: "
                    "`python -m src.cleanOutputs` then `python -m src.runInference`)."
                )
            else:
                selectedRuleIndex = st.selectbox(
                    "Choose Rule",
                    range(len(entries)),
                    format_func=lambda i: titleByName.get(
                        str(entries[i].get("ruleName", "")),
                        entries[i].get("ruleName", i),
                    ),
                )
                transcriptEntry = entries[selectedRuleIndex]
                with st.expander("User Prompt", expanded=False):
                    st.text(transcriptEntry.get("openAiUserPrompt", "")[:12000])
                with st.expander("Assistant Reply", expanded=False):
                    st.text(transcriptEntry.get("openAiAssistantReply", "")[:8000])

    with tabInfer:
        # inference_result.json: metrics, inferred edge blocks, embedded trace sample rows.
        inferenceDoc = loadJson(inferencePath)
        if not isinstance(inferenceDoc, dict):
            st.warning("No `inference_result.json`. Run `python -m src.runInference`.")
        else:
            st.metric("Rule Trace Edge Records", inferenceDoc.get("ruleTraceEdgeRecordCount", "—"))
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("LLM Generation")
                st.json(inferenceDoc.get("llmGeneration", {}))
            with col2:
                st.subheader("Paths")
                st.json(
                    {
                        k: v
                        for k, v in inferenceDoc.items()
                        if k.endswith("RelativePath") or k == "graphmlRelativePath"
                    }
                )
            watchableBlocks = inferenceDoc.get("inferredEdgesWatchable")
            if isinstance(watchableBlocks, list) and watchableBlocks:
                st.subheader("Inferred Edges By Rule")
                st.caption(
                    "Each block is one head predicate; titles match **display_title** in **rules.json**. "
                    "Re-run **python -m src.runInference** if this section is missing."
                )
                for block in watchableBlocks:
                    titles = block.get("watchableRuleTitles") or []
                    ruleNames = block.get("ruleNames") or []
                    pred = str(block.get("headPredicate", "") or "")
                    titleLine = " · ".join(str(t) for t in titles) if titles else pred
                    ruleNameList = ruleNames if isinstance(ruleNames, list) else [ruleNames]
                    ruleNamesJoined = ", ".join(str(x) for x in ruleNameList) if ruleNameList else ""
                    st.markdown(
                        f"**{titleLine}**  \n"
                        f"**headPredicate:** {pred} · **ruleNames:** {ruleNamesJoined}"
                    )
                    rows = block.get("edges") or []
                    if rows:
                        edgesFrame = pd.DataFrame(rows)
                        edgesFrame.insert(0, "watchableRuleTitles", [titleLine] * len(edgesFrame))
                        st.dataframe(edgesFrame, use_container_width=True, hide_index=True)
                    else:
                        st.caption("(No Edges)")
            else:
                predTitles = rule_display.predicateToWatchableTitles(showableList)
                inferred = inferenceDoc.get("inferredEdgesByPredicate")
                if isinstance(inferred, dict):
                    st.subheader("Inferred Head Edges (Last Timestep — Legacy Shape)")
                    for pred, rows in inferred.items():
                        wtitle = predTitles.get(str(pred), "")
                        header = f"**{wtitle}** ({pred})" if wtitle else f"**{pred}**"
                        st.markdown(header)
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        else:
                            st.caption("(none)")
            traceSampleRows = inferenceDoc.get("ruleTraceEdgeSample")
            if isinstance(traceSampleRows, list) and traceSampleRows:
                st.subheader("Trace Samples")
                st.caption(
                    "First rows from the run log. From rule shows your display_title when Triggered by is Rule; "
                    "previously known fact when PyReason only loaded a grounding edge (no rule name applies)."
                )
                sampleFrame = pd.DataFrame(traceSampleRows)
                if "label" in sampleFrame.columns and "occurredDueTo" in sampleFrame.columns:
                    sampleFrame = addRuleTitleColumn(
                        sampleFrame,
                        showableList,
                        labelColumn="label",
                        triggeredColumn="occurredDueTo",
                    )
                sampleFrame = dropColumnsIfPresent(sampleFrame, INFER_SAMPLE_DROP)
                st.dataframe(sampleFrame, use_container_width=True, hide_index=True)

    with tabTrace:
        # Full PyReason trace is CSV; we show a capped preview and the same From Rule helper column as infer tab.
        if not traceCsv.is_file():
            st.warning("No `inference_rule_trace_edges.csv`. Run inference first.")
        else:
            st.markdown(
                """
### What this trace is

PyReason wrote this file while reason() ran. Each row is one update to a labeled edge truth interval (not a copy of GraphML or the LLM fact list).

- **Time**: simulation timestep (0, 1, ...). Rules can fire across steps.
- **Fixed-Point-Operation**: inside that timestep, PyReason repeats rule application until nothing new appears (or hits an internal limit). This counts which pass that was (0 = first pass at that timestep).
- **Edge / Label**: tuple (source, target) and the predicate on that directed edge.
- **New bound**: fuzzy truth interval. [1.0, 1.0] means fully true (both ends 1; interval form, not a ratio 1:1).
- **Triggered by**: Fact = loaded grounding edge; Rule = produced by a PyReason rule (see From rule).
- **From rule**: for Rule rows, your display_title from rules.json. For Fact rows, previously known fact (loaded, not inferred by a rule).
- **Row count**: every edge update PyReason logged across timesteps.

Columns Old bound, Occurred due to, and Inconsistency message are hidden below when empty or noisy; open the CSV if you need them.
"""
            )
            traceFrame = pd.read_csv(traceCsv)
            traceFrame = dropColumnsIfPresent(traceFrame, TRACE_COLUMNS_DROP)
            if "Label" in traceFrame.columns and "Triggered By" in traceFrame.columns:
                traceFrame = addRuleTitleColumn(
                    traceFrame,
                    showableList,
                    labelColumn="Label",
                    triggeredColumn="Triggered By",
                )
            st.markdown("##### Trace Samples")
            st.caption(
                "From rule matches display_title when Triggered by is Rule; previously known fact when Triggered by is Fact."
            )
            st.dataframe(traceFrame.head(500), use_container_width=True, height=400)
            st.caption(
                f"First 500 of {len(traceFrame)} rows. Full file: {traceCsv.relative_to(_ROOT)}."
            )


if __name__ == "__main__":
    main()
