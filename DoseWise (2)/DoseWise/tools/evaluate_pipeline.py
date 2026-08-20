"""Deterministic evaluation for the structured DoseWise pipeline.

Run from the project root with: py -3.14 tools/evaluate_pipeline.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / 'data' / 'evaluation_cases.json'
sys.path.insert(0, str(ROOT))

from src.rag_pipeline import process_message
from src.drug_resolver import normalize_text


def _normal_set(values: Iterable[str]) -> set:
    return {normalize_text(value) for value in values if value}


def _pair_key(pair: Iterable[str]) -> Tuple[str, str]:
    return tuple(sorted(normalize_text(value) for value in pair))


def _actual_interaction_pairs(retrievals: List[Dict[str, Any]]) -> set:
    return {
        _pair_key((item['drug_a'], item['drug_b']))
        for item in retrievals
        if item.get('ddinter_rows', 0) > 0 and item.get('interaction_level')
    }


def _citation_is_auditable(item: Dict[str, Any]) -> bool:
    if item.get('source') != 'DDInter':
        return False
    if item.get('ddinter_rows', 0) == 0:
        return not item.get('chunk_ids') and not item.get('source_rows')
    records = item.get('matches') or []
    return bool(records) and all(
        record.get('chunk_id')
        and record.get('source_row')
        and record.get('evidence') == f"DDInter master_interactions.csv row {record['source_row']}"
        for record in records
    )


def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    result = process_message(case['message'])
    actual_generics = _normal_set(result.get('verified_generics', []))
    expected_generics = _normal_set(case.get('expected_verified_generics', []))
    actual_pairs = _actual_interaction_pairs(result.get('retrievals', []))
    expected_pairs = {_pair_key(pair) for pair in case.get('expected_interaction_pairs', [])}
    clarification = any(not item.get('verified') for item in result.get('resolved', []))
    citation_accuracy = all(_citation_is_auditable(item) for item in result.get('retrievals', []))
    return {
        'name': case['name'],
        'resolution_correct': actual_generics == expected_generics,
        'retrieval_correct': actual_pairs == expected_pairs,
        'citation_correct': citation_accuracy,
        'clarification_correct': clarification == case.get('expected_clarification', False),
        'actual_verified_generics': sorted(actual_generics),
        'actual_interaction_pairs': sorted(actual_pairs),
    }


def run_evaluation(cases: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    if cases is None:
        cases = json.loads(CASES_PATH.read_text(encoding='utf-8'))
    results = [evaluate_case(case) for case in cases]
    total = len(results)

    def rate(key: str) -> float:
        return round(sum(item[key] for item in results) / total, 3) if total else 0.0

    return {
        'cases': total,
        'resolution_accuracy': rate('resolution_correct'),
        'retrieval_accuracy': rate('retrieval_correct'),
        'citation_accuracy': rate('citation_correct'),
        'clarification_accuracy': rate('clarification_correct'),
        'precision_at_k': 'not_applicable: deterministic exact-pair lookup returns all matching records, not a ranked list',
        'recall': 'not_applicable: no ranked candidate retrieval or corpus-level relevance set',
        'unsupported_claim_rate': 'not_applicable: no LLM generation layer; responses are deterministic and retrieval-grounded',
        'results': results,
    }


if __name__ == '__main__':
    summary = run_evaluation()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
