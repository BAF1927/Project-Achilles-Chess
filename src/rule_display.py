# Helpers for human-readable rule names in the dashboard and for inference_result.json blocks.
# headPredicateFromRuleText reads the predicate in the rule head (text left of PyReason "<-" or "<-1").
from __future__ import annotations

import re
from typing import Any


def displayTitleForRule(rule: dict[str, Any]) -> str:
    # Prefer explicit display_title in rules.json; otherwise split CamelCase rule name into words.
    explicit = rule.get("display_title")
    if explicit:
        return str(explicit)
    name = str(rule.get("name") or "Rule")
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name)


def headPredicateFromRuleText(ruleText: str) -> str | None:
    # Example: "citizenOf(X,Y) <- ..." returns "citizenOf".
    text = ruleText.strip()
    if "<-1" in text:
        headPart = text.split("<-1", 1)[0].strip()
    elif "<-" in text:
        headPart = text.split("<-", 1)[0].strip()
    else:
        return None
    match = re.match(r"^(\w+)\s*\(", headPart)
    return match.group(1) if match else None


def predicateToWatchableTitles(showableRules: list[dict[str, Any]]) -> dict[str, str]:
    # Maps inferred head predicate string to one or more display_title values joined for tables.
    bucket: dict[str, list[str]] = {}
    for rule in showableRules:
        rt = str(rule.get("rule_text", ""))
        pred = headPredicateFromRuleText(rt)
        if not pred:
            continue
        title = displayTitleForRule(rule)
        bucket.setdefault(pred, [])
        if title not in bucket[pred]:
            bucket[pred].append(title)
    return {p: " · ".join(titles) for p, titles in bucket.items()}


def predRuleMetadata(showableRules: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    # Internal: for each head predicate, collect watchableRuleTitles and rule name ids.
    meta: dict[str, dict[str, list[str]]] = {}
    for rule in showableRules:
        pred = headPredicateFromRuleText(str(rule.get("rule_text", "")))
        if not pred:
            continue
        slot = meta.setdefault(pred, {"watchableRuleTitles": [], "ruleNames": []})
        t = displayTitleForRule(rule)
        n = rule.get("name")
        if t not in slot["watchableRuleTitles"]:
            slot["watchableRuleTitles"].append(t)
        if n and str(n) not in slot["ruleNames"]:
            slot["ruleNames"].append(str(n))
    return meta


def inferredEdgesWatchableList(
    inferredByPredicate: dict[str, list[dict[str, Any]]],
    showableRules: list[dict[str, Any]],
    *,
    predicateOrder: list[str] | None = None,
) -> list[dict[str, Any]]:
    # One block per head predicate: watchable titles, rule ids, edges (same as inferredEdgesByPredicate).
    predMeta = predRuleMetadata(showableRules)
    keys = predicateOrder if predicateOrder else sorted(inferredByPredicate.keys())
    out: list[dict[str, Any]] = []
    for pred in keys:
        if pred not in inferredByPredicate:
            continue
        edges = inferredByPredicate[pred]
        meta = predMeta.get(pred, {})
        out.append(
            {
                "headPredicate": pred,
                "watchableRuleTitles": meta.get("watchableRuleTitles", []),
                "ruleNames": meta.get("ruleNames", []),
                "edgeCount": len(edges),
                "edges": edges,
            }
        )
    return out
