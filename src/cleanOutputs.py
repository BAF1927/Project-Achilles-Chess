# Deletes listed artifact files under src/output so the next runInference starts from a clean slate.
# Never deletes rules.json or yago_subset.graphml in src/data. Rebuild outputs with: python -m src.runInference
from __future__ import annotations
from . import paths

# Filenames that runInference or llmClient create under writableOutputDirectory.
GENERATED_NAMES = (
    paths.inferenceResultFileName,
    paths.openAiTranscriptFileName,
    "llm_openai_responses.json",
    "llm_facts_pyreason.json",
    "llm_full_payload.json",
    "inference_rule_trace_edges.csv",
    "rules_for_pyreason_loader.json",
)


def cleanOutputDirectory() -> list[str]:
    # Returns repo-relative paths of files removed (empty list if nothing matched).
    removedRelativePaths: list[str] = []
    root = paths.repositoryRootDirectory
    for name in GENERATED_NAMES:
        candidate = paths.writableOutputDirectory / name
        if candidate.is_file():
            candidate.unlink()
            try:
                removedRelativePaths.append(str(candidate.relative_to(root)))
            except ValueError:
                removedRelativePaths.append(str(candidate))
    return removedRelativePaths


if __name__ == "__main__":
    gone = cleanOutputDirectory()
    if gone:
        for line in gone:
            print(f"removed {line}")
    else:
        print("nothing to remove under src/output/")
    print(
        "Unchanged: src/data/rules.json, src/data/yago_subset.graphml "
        "(run buildYagoSubsetFromHuggingFace only if you need a fresh YAGO GraphML)."
    )
