# Downloads YAGO3-10 via Hugging Face datasets in streaming mode until a fixed set of triples is seen.
# YAGO relation strings (isLocatedIn, hasOfficialLanguage) become GraphML edge keys used in rules.json.
from __future__ import annotations
from pathlib import Path
from typing import Any
import networkx as nx
from . import paths

huggingFaceDatasetName = "VLyB/YAGO3-10"


def yagoRelationToPyreasonEdgeKey(headEntity: str, relation: str, tailEntity: str) -> str:
    # Maps dataset relation names to PyReason attribute keys on DiGraph edges.
    if relation == "isLocatedIn":
        if headEntity == "Google" and tailEntity == "United_States":
            return "companyHeadquarteredIn"
        return "cityIn"
    if relation == "hasOfficialLanguage":
        return "officialLanguage"
    raise ValueError(f"No PyReason mapping for relation {relation!r}")


def requiredYagoTriples() -> set[tuple[str, str, str]]:
    # Minimal triples the assignment graph must contain before we stop streaming the dataset.
    return {
        ("Hartlepool", "isLocatedIn", "England"),
        ("London", "isLocatedIn", "United_Kingdom"),
        ("Madrid", "isLocatedIn", "Spain"),
        ("England", "hasOfficialLanguage", "English_language"),
        ("Spain", "hasOfficialLanguage", "Spanish_language"),
        ("Google", "isLocatedIn", "United_States"),
    }


def buildKnowledgeGraphFromYagoStream(yagoDatasetStream: Any) -> nx.DiGraph:
    # Consumes streaming rows until every required triple is matched or stream ends with an error.
    pendingTriples = requiredYagoTriples()
    knowledgeGraph = nx.DiGraph()
    for row in yagoDatasetStream:
        key = (row["head"], row["relation"], row["tail"])
        if key not in pendingTriples:
            continue
        edgeKey = yagoRelationToPyreasonEdgeKey(row["head"], row["relation"], row["tail"])
        knowledgeGraph.add_edge(row["head"], row["tail"], **{edgeKey: 1})
        pendingTriples.remove(key)
        if not pendingTriples:
            break
    if pendingTriples:
        missingList = ", ".join(str(item) for item in sorted(pendingTriples))
        raise RuntimeError(f"Could not find all YAGO triples before end of stream. Still missing: {missingList}")
    return knowledgeGraph


def writeGraphmlFromHuggingFace(destinationPath: Path | None = None) -> Path:
    # Entry for CLI: stream dataset, build nx graph, write_graphml to paths.graphmlPath() by default.
    from datasets import load_dataset

    targetPath = destinationPath or paths.graphmlPath()
    targetPath.parent.mkdir(parents=True, exist_ok=True)
    trainStream = load_dataset(huggingFaceDatasetName, split="train", streaming=True)
    graph = buildKnowledgeGraphFromYagoStream(trainStream)
    nx.write_graphml(graph, targetPath)
    return targetPath


if __name__ == "__main__":
    writtenPath = writeGraphmlFromHuggingFace()
    print(f"Wrote {writtenPath}")
