# Leibniz Lab - YAGO and PyReason Assessment

This project combines a **YAGO knowledge graph subset**, **LLM-generated grounding facts**, and **PyReason inference**.  
The goal is to show that explicit rules can fire on top of trusted graph facts plus new facts suggested by an LLM.

## Project Overview

- **Knowledge graph input:** `src/data/yago_subset.graphml`
- **Rules input:** `src/data/rules.json` (five PyReason-compatible rules)
- **LLM grounding output:** `src/output/llm_facts_pyreason.json`
- **Inference output:** `src/output/inference_result.json` and `src/output/inference_rule_trace_edges.csv`
- **Human summary:** `src/FINDINGS.txt`

The dashboard overlays three layers in one graph view:
- **YAGO edges** (solid)
- **LLM fact edges** (dashed)
- **Inferred rule-head edges** (thick)

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` before running inference.

## Data and Inference Flow

### 1) Build the YAGO GraphML subset

```bash
python -m src.buildYagoSubsetFromHuggingFace
```

**How this works:** `buildYagoSubsetFromHuggingFace` streams `VLyB/YAGO3-10` and keeps a small fixed set of required triples.  
It writes only that minimal subset to `src/data/yago_subset.graphml`.

### 2) Run the full pipeline (LLM + PyReason)

```bash
python -m src.runInference
```

This step:
- reads `yago_subset.graphml` and `rules.json`
- calls OpenAI once per showable rule
- converts model output into PyReason fact JSON
- runs `reason()`
- writes inference artifacts under `src/output/`

## Dashboard

Run:

```bash
./run_dashboard.sh
```

The dashboard is read-only and loads files from disk.

- **Default mode:** reads `src/output/`
- **Demo mode:** reads frozen sample artifacts in `src/brunoOutput/`

Demo mode command:

```bash
LEIBNIZ_USE_BRUNO_OUTPUT=1 ./run_dashboard.sh
```

`runInference` always writes to `src/output` only.  
It does not modify `src/brunoOutput`.

## Reset Generated Outputs

```bash
python -m src.cleanOutputs
```

This removes generated files under `src/output` and keeps:
- `src/data/rules.json`
- `src/data/yago_subset.graphml`

## Useful Output Files

- **LLM prompt/reply audit:** `src/output/llm_openai_prompts_and_replies.json`
- **Merged LLM payload:** `src/output/llm_full_payload.json`
- **PyReason fact file:** `src/output/llm_facts_pyreason.json`
- **Inference bundle:** `src/output/inference_result.json`
- **Full trace CSV:** `src/output/inference_rule_trace_edges.csv`

## macOS SSL Note

If OpenAI calls fail with `CERTIFICATE_VERIFY_FAILED`, install dependencies again and retry:

```bash
pip install -r requirements.txt
python -m src.runInference
```
