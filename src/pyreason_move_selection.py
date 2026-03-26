# Builds a one-shot PyReason graph per half-move: each legal UCI string is a node, facts
# attach interval scores, and Ranked(x) rules combine them. Python never re-orders moves
# except by reading the interpretation trace lower bounds.
from __future__ import annotations

import contextlib
import io
from typing import Dict, List, Tuple

import chess
import networkx as nx

moveRankingRulesHaveBeenInstalled = False
pyreasonFactIdentifierCounter = 0


# Registers Ranked(x) <- Predicate(x) rules exactly once per interpreter lifetime.
def installMoveRankRules(pyreasonModule) -> None:
    global moveRankingRulesHaveBeenInstalled
    if moveRankingRulesHaveBeenInstalled:
        return
    pyreasonModule.add_rule(pyreasonModule.Rule("Ranked(x)<-Capture(x)", "rank_from_capture"))
    pyreasonModule.add_rule(pyreasonModule.Rule("Ranked(x)<-Check(x)", "rank_from_check"))
    pyreasonModule.add_rule(pyreasonModule.Rule("Ranked(x)<-Center(x)", "rank_from_center"))
    pyreasonModule.add_rule(pyreasonModule.Rule("Ranked(x)<-Safe(x)", "rank_from_safe"))
    pyreasonModule.add_rule(pyreasonModule.Rule("Ranked(x)<-TradeUp(x)", "rank_from_tradeup"))
    pyreasonModule.add_rule(pyreasonModule.Rule("Ranked(x)<-PinOk(x)", "rank_from_pinok"))
    pyreasonModule.add_rule(pyreasonModule.Rule("Ranked(x)<-KingPress(x)", "rank_from_kingpress"))
    pyreasonModule.add_rule(pyreasonModule.Rule("Ranked(x)<-Cramp(x)", "rank_from_cramp"))
    pyreasonModule.add_rule(pyreasonModule.Rule("Ranked(x)<-Develop(x)", "rank_from_develop"))
    pyreasonModule.add_rule(pyreasonModule.Rule("Ranked(x)<-UnderFire(x)", "rank_from_underfire"))
    pyreasonModule.add_rule(pyreasonModule.Rule("Ranked(x)<-QValue(x)", "rank_from_qvalue"))
    moveRankingRulesHaveBeenInstalled = True


def bootstrapPyreason(pyreasonModule) -> None:
    pyreasonModule.settings.verbose = False
    pyreasonModule.settings.store_interpretation_changes = True
    pyreasonModule.settings.atom_trace = False
    with contextlib.redirect_stdout(io.StringIO()):
        pyreasonModule.load_graph(nx.DiGraph())
    installMoveRankRules(pyreasonModule)


def runPyreasonForMoves(
    pyreasonModule,
    legalMoves: List[chess.Move],
    uciToFacts: Dict[str, Dict[str, Tuple[float, float]]],
) -> Tuple[Dict[str, float], str]:
    global pyreasonFactIdentifierCounter
    with contextlib.redirect_stdout(io.StringIO()):
        reasoningGraph = nx.DiGraph()
        for legalMove in legalMoves:
            reasoningGraph.add_node(legalMove.uci())
        pyreasonModule.load_graph(reasoningGraph)
        installMoveRankRules(pyreasonModule)

        for legalMove in legalMoves:
            uciString = legalMove.uci()
            predicateBounds = uciToFacts[uciString]
            for predicateName, (lowBound, highBound) in predicateBounds.items():
                lowBound = max(0.01, min(0.99, float(lowBound)))
                highBound = max(0.01, min(0.99, float(highBound)))
                if lowBound > highBound:
                    lowBound, highBound = highBound, lowBound
                pyreasonFactIdentifierCounter += 1
                factLabel = f"{predicateName}({uciString}):[{lowBound},{highBound}]"
                factIdentifier = f"f_{pyreasonFactIdentifierCounter}_{predicateName}_{uciString}"
                pyreasonModule.add_fact(pyreasonModule.Fact(factLabel, factIdentifier))

        interpretation = pyreasonModule.reason(
            timesteps=2,
            convergence_threshold=-1,
            convergence_bound_threshold=-1,
            queries=None,
            again=False,
            restart=True,
        )

    rankedLowerBounds: Dict[str, float] = {}
    predicateLowerBoundsPerMove: Dict[str, Dict[str, float]] = {uciKey: {} for uciKey in uciToFacts.keys()}
    for traceRow in interpretation.rule_trace_node:
        _timestep, _fingerprint, componentUci, labelAtom, boundInterval, *_unused = traceRow
        predicateName = labelAtom.get_value()
        if not isinstance(componentUci, str):
            continue
        if componentUci not in uciToFacts:
            continue
        lowerNumeric = float(boundInterval.lower)
        if predicateName == "Ranked":
            rankedLowerBounds[componentUci] = max(rankedLowerBounds.get(componentUci, 0.0), lowerNumeric)
        elif predicateName in uciToFacts[componentUci]:
            predicateLowerBoundsPerMove[componentUci][predicateName] = max(
                predicateLowerBoundsPerMove[componentUci].get(predicateName, 0.0), lowerNumeric
            )

    if not rankedLowerBounds:
        for uciString, predicateScores in predicateLowerBoundsPerMove.items():
            if not predicateScores:
                rankedLowerBounds[uciString] = 0.0
            else:
                rankedLowerBounds[uciString] = sum(predicateScores.values()) / float(len(predicateScores))

    inferenceSummary = f"PyReason | atoms={len(interpretation.rule_trace_node)} | uci-scored={len(rankedLowerBounds)}"
    return rankedLowerBounds, inferenceSummary


def pickUciFromPyreasonScores(uciToScore: Dict[str, float]) -> str | None:
    if not uciToScore:
        return None
    return max(uciToScore.keys(), key=lambda uciString: (uciToScore[uciString], uciString))
