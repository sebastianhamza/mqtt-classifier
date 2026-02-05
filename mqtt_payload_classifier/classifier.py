"""Combines heuristic rules with the semantic classifier."""

import re
from collections import namedtuple
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count

from .constants import Category
from .heuristic import (
    detect_structure,
    heuristic_classify_multi,
    detect_sensitive,
    compute_risk,
)
from .semantic import (
    semantic_classify_multi,
    load_spacy_model,
    build_category_vectors,
)

_SEM_NLP = None
_SEM_VECTORS = {}
_SEM_ENABLED = False
try:
    _SEM_NLP, _has = load_spacy_model()
    if _has:
        _SEM_VECTORS = build_category_vectors(_SEM_NLP)
        _SEM_ENABLED = bool(_SEM_VECTORS)
except Exception:
    _SEM_NLP = None
    _SEM_VECTORS = {}
    _SEM_ENABLED = False


# Result structure
Result = namedtuple(
    "Result",
    [
        "structure",
        "categories",
        "category_scores",
        "risk_flags",
        "risk_score",
        "decision_strategy",
        "heuristic_categories",
        "semantic_categories",
        "payload",
    ],
)


def _merge_multilabel_predictions(
    heuristic_scores: Dict[Category, float],
    semantic_scores: Dict[Category, float],
    heur_threshold: float = 0.3,
    sem_threshold: float = 0.15,
    soft_diff_threshold: float = 0.15,
) -> Tuple[Dict[Category, float], str]:
    """
    Merge heuristic and semantic scores using confidence-weighted averaging.
    Semantic scores are probabilities (sum to 1) so they're lower magnitude
    than heuristic ones. Each method is weighted by its confidence, and
    secondary categories get dropped when the top one dominates.

    Returns (merged_scores, decision_strategy).
    """
    merged: Dict[Category, float] = {}
    
    # lower threshold for semantic since its scores are probabilities
    heur_cats = {cat: score for cat, score in heuristic_scores.items() if score >= heur_threshold}
    sem_cats = {cat: score for cat, score in semantic_scores.items() if score >= sem_threshold}
    
    # pick strategy
    if heur_cats and sem_cats:
        strategy = "hybrid"
    elif heur_cats:
        strategy = "heuristic"
    elif sem_cats:
        strategy = "semantic"
    else:
        strategy = "none"
    
    # adaptive weighting based on confidence
    heur_max_conf = max(heuristic_scores.values()) if heuristic_scores else 0.0
    sem_max_conf = max(semantic_scores.values()) if semantic_scores else 0.0
    
    heur_confidence = min(heur_max_conf, 1.0) if heur_max_conf > 0 else 0.0
    sem_confidence = min(sem_max_conf * 10, 1.0) if sem_max_conf > 0 else 0.0
    
    total_conf = heur_confidence + sem_confidence
    if total_conf > 0:
        heur_weight = heur_confidence / total_conf
        sem_weight = sem_confidence / total_conf
    else:
        heur_weight = 0.5
        sem_weight = 0.5
    
    all_cats = set(heur_cats.keys()) | set(sem_cats.keys())
    
    for cat in all_cats:
        h_score = heur_cats.get(cat, 0.0)
        s_score = sem_cats.get(cat, 0.0)
        
        if cat in heur_cats and cat in sem_cats:
            # both methods agree: weighted average
            merged[cat] = round(h_score * heur_weight + s_score * sem_weight, 3)
        elif cat in heur_cats:
            # Heuristic only
            merged[cat] = round(h_score, 3)
        else:
            # Semantic only
            merged[cat] = round(s_score, 3)
    
    # drop secondary categories if the top one is way ahead
    if merged:
        sorted_cats = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_cats) >= 2:
            top_score = sorted_cats[0][1]
            
            if top_score > 0.5:
                filtered = {cat: score for cat, score in sorted_cats if (top_score - score) <= soft_diff_threshold}
                if filtered:
                    merged = dict(filtered)
    
    return merged, strategy


def classify_payload(payload: str, sanitize: bool = True) -> Result:
    """Run multilabel classification on a single payload."""
    MAX_PAYLOAD_SIZE = 5000
    original_payload = payload
    if len(payload) > MAX_PAYLOAD_SIZE:
        payload = payload[:MAX_PAYLOAD_SIZE]
    
    struct = detect_structure(payload)
    flags = detect_sensitive(payload)
    risk = compute_risk(flags)
    
    heuristic_scores = heuristic_classify_multi(payload, struct)
    
    semantic_scores: Dict[Category, float] = {}
    if _SEM_ENABLED:
        semantic_scores = semantic_classify_multi(payload, _SEM_NLP, _SEM_VECTORS)
    
    category_scores, decision_strategy = _merge_multilabel_predictions(
        heuristic_scores, semantic_scores
    )
    
    categories = sorted(
        [cat for cat, score in category_scores.items() if score >= 0.15],
        key=lambda cat: category_scores[cat],
        reverse=True,
    )
    if not categories:
        category_scores = {Category.UNSTRUCTURED: 0.0}
        categories = [Category.UNSTRUCTURED]
        decision_strategy = "none"
    
    return Result(
        structure=struct,
        categories=categories,
        category_scores=category_scores,
        risk_flags=flags,
        risk_score=risk,
        decision_strategy=decision_strategy,
        heuristic_categories=heuristic_scores,
        semantic_categories=semantic_scores,
        payload=original_payload if not sanitize else _sanitize_payload(original_payload),
    )


def _sanitize_payload(payload: str) -> str:
    """Mask long numbers in payload."""
    return re.sub(
        r"\b(\d{6,})\b",
        lambda m: m.group(1)[:3] + "..." + m.group(1)[-3:],
        payload.replace("\n", " \\n "),
    )


def _classify_payload_wrapper(payload: str) -> Result:
    """Wrapper for picklable multiprocessing."""
    return classify_payload(payload, sanitize=False)


def classify_many(payloads, sanitize: bool = True, batch_size: int = 100, show_progress: bool = False):
    results = []
    total = len(payloads)
    
    for i, payload in enumerate(payloads):
        results.append(classify_payload(payload, sanitize=sanitize))
        
        if show_progress and (i + 1) % batch_size == 0:
            print(f"  Processed {i + 1}/{total} payloads ({((i + 1) / total * 100):.1f}%)", flush=True)
    
    if show_progress and total % batch_size != 0:
        print(f"  Processed {total}/{total} payloads (100.0%)", flush=True)
    
    return results


def classify_many_parallel(payloads, sanitize: bool = True, num_workers: int = None, show_progress: bool = False):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    if len(payloads) < 100:
        return classify_many(payloads, sanitize=sanitize, show_progress=show_progress)
    
    results = []
    processed = 0
    
    try:
        with Pool(processes=num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_classify_payload_wrapper, payloads, chunksize=50)):
                if sanitize:
                    result = Result(
                        structure=result.structure,
                        categories=result.categories,
                        category_scores=result.category_scores,
                        risk_flags=result.risk_flags,
                        risk_score=result.risk_score,
                        decision_strategy=result.decision_strategy,
                        heuristic_categories=result.heuristic_categories,
                        semantic_categories=result.semantic_categories,
                        payload=_sanitize_payload(result.payload),
                    )
                
                results.append(result)
                processed = i + 1
                
                if show_progress and processed % 500 == 0:
                    print(f"  Processed {processed}/{len(payloads)} payloads ({(processed / len(payloads) * 100):.1f}%)", flush=True)
        
        if show_progress and len(payloads) % 500 != 0:
            print(f"  Processed {len(payloads)}/{len(payloads)} payloads (100.0%)", flush=True)
    except Exception as e:
        print(f"Parallel processing failed ({e}), falling back to serial processing", flush=True)
        return classify_many(payloads, sanitize=sanitize, show_progress=show_progress)
    
    return results
