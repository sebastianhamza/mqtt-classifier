import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mqtt_payload_classifier import classify_payload
from mqtt_payload_classifier.constants import Category, RiskFlag


def load_test_cases(json_file: str) -> List[Dict]:
    with open(json_file, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    return data.get('test_cases', [])


def evaluate_single_label(test_cases: List[Dict], verbose: bool = False) -> Dict:
    results = []
    cat_correct = 0
    flags_correct = 0
    both_correct = 0
    total_flag_score = 0.0

    for case in test_cases:
        if 'multi_label_categories' in case:
            continue
        
        payload = case['payload']
        expected_cat = case['expected_category']
        expected_flags = set(case.get('expected_flags', []))
        
        try:
            result = classify_payload(payload)
            got_cat = result.categories[0] if result.categories else Category.UNSTRUCTURED
            got_flags = set(f.value if hasattr(f, 'value') else str(f) for f in (result.risk_flags or []))
        except Exception as e:
            if verbose:
                print(f"ERROR processing {case['name']}: {e}")
            got_cat = Category.UNSTRUCTURED
            got_flags = set()
        
        ok_cat = (got_cat == expected_cat)
        
        if len(expected_flags) == 0:
            ok_flags = (len(got_flags) == 0)
            flag_score = 1.0 if ok_flags else 0.0
        else:
            matched = len(expected_flags.intersection(got_flags))
            flag_score = matched / len(expected_flags)
            ok_flags = (matched == len(expected_flags))
        
        total_flag_score += flag_score
        
        if ok_cat:
            cat_correct += 1
        if ok_flags:
            flags_correct += 1
        if ok_cat and ok_flags:
            both_correct += 1
        
        results.append({
            "id": case.get('id', 'unknown'),
            "name": case['name'],
            "expected_category": expected_cat,
            "got_category": got_cat.value if got_cat else None,
            "expected_flags": sorted(list(expected_flags)),
            "got_flags": sorted(list(got_flags)),
            "category_ok": ok_cat,
            "flags_ok": ok_flags,
            "flag_score": round(flag_score, 3),
            "structure": result.structure,
            "risk_score": result.risk_score,
            "decision_strategy": result.decision_strategy,
        })
    
    total = len([c for c in test_cases if 'multi_label_categories' not in c])
    
    summary = {
        "test_type": "single_label",
        "total": total,
        "category_accuracy": round(cat_correct / total, 3) if total > 0 else 0,
        "flags_accuracy": round(flags_correct / total, 3) if total > 0 else 0,
        "both_accuracy": round(both_correct / total, 3) if total > 0 else 0,
        "avg_flag_match_score": round(total_flag_score / total, 3) if total > 0 else 0,
    }
    
    return {
        "results": results,
        "summary": summary,
    }


def evaluate_multi_label(test_cases: List[Dict], verbose: bool = False) -> Dict:
    results = []
    set_correct = 0
    rank_correct = 0
    partial_correct = 0
    
    multilabel_cases = [c for c in test_cases if 'multi_label_categories' in c]
    
    if not multilabel_cases:
        return {
            "results": [],
            "summary": {
                "test_type": "multi_label",
                "total": 0,
                "note": "No multi-label test cases found",
            }
        }
    
    for case in multilabel_cases:
        payload = case['payload']
        expected_cats = set(case['multi_label_categories'])
        expected_scores = case.get('multi_label_confidence', {})
        
        try:
            result = classify_payload(payload)
            got_cats = set(result.categories)
            got_scores = result.category_scores
            
        except Exception as e:
            if verbose:
                print(f"ERROR processing {case['name']}: {e}")
            got_cats = set()
            got_scores = {}
        
        ok_set = (got_cats == expected_cats)
        ok_rank = expected_cats.issubset(got_cats)
        ok_partial = len(expected_cats.intersection(got_cats)) > 0
        
        if ok_set:
            set_correct += 1
        if ok_rank:
            rank_correct += 1
        if ok_partial:
            partial_correct += 1
        
        results.append({
            "id": case.get('id', 'unknown'),
            "name": case['name'],
            "expected_categories": sorted(list(expected_cats)),
            "got_categories": sorted([c.value if hasattr(c, 'value') else str(c) for c in got_cats]),
            "set_ok": ok_set,
            "rank_ok": ok_rank,
            "partial_ok": ok_partial,
            "category_scores": {k.value if hasattr(k, 'value') else str(k): v for k, v in got_scores.items()},
            "structure": result.structure,
            "risk_score": result.risk_score,
            "decision_strategy": result.decision_strategy,
        })
    
    total = len(multilabel_cases)
    
    summary = {
        "test_type": "multi_label",
        "total": total,
        "set_accuracy": round(set_correct / total, 3) if total > 0 else 0,
        "rank_accuracy": round(rank_correct / total, 3) if total > 0 else 0,
        "partial_accuracy": round(partial_correct / total, 3) if total > 0 else 0,
    }
    
    return {
        "results": results,
        "summary": summary,
    }


def evaluate_by_category(test_cases: List[Dict]) -> Dict:
    categories = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for case in test_cases:
        if 'multi_label_categories' in case:
            continue
        
        expected_cat = case['expected_category']
        categories[expected_cat]["total"] += 1
        
        try:
            result = classify_payload(case['payload'])
            got_cat = result.categories[0] if result.categories else Category.UNSTRUCTURED
            if got_cat == expected_cat:
                categories[expected_cat]["correct"] += 1
        except:
            pass
    
    breakdown = {}
    for cat, stats in categories.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        breakdown[cat] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": round(accuracy, 3),
        }
    
    return breakdown


def build_confusion_matrix(test_cases: List[Dict]) -> Tuple[Dict, Dict]:
    categories = set()
    for case in test_cases:
        if 'multi_label_categories' not in case:
            categories.add(case['expected_category'])
    
    categories = sorted(list(categories))
    confusion = {cat: {pred: 0 for pred in categories} for cat in categories}
    
    for case in test_cases:
        if 'multi_label_categories' in case:
            continue
        
        expected_cat = case['expected_category']
        try:
            result = classify_payload(case['payload'])
            got_cat = result.categories[0] if result.categories else 'unstructured'
            if isinstance(got_cat, Category):
                got_cat = got_cat.value
            confusion[expected_cat][got_cat] = confusion[expected_cat].get(got_cat, 0) + 1
        except:
            pass
    
    metrics = {}
    for cat in categories:
        tp = confusion[cat][cat]
        fp = sum(confusion[other][cat] for other in categories if other != cat)
        fn = sum(confusion[cat][other] for other in categories if other != cat)
        support = sum(confusion[cat].values())
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[cat] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    
    return confusion, metrics


def print_results(evaluation: Dict, test_type: str = "single_label"):
    summary = evaluation.get("summary", {})
    results = evaluation.get("results", [])
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {test_type.upper()}")
    print(f"{'='*80}\n")
    
    print("SUMMARY METRICS")
    print("-" * 80)
    for key, value in summary.items():
        if key != "test_type" and key != "note":
            print(f"  {key:.<30} {value}")
    
    if "note" in summary:
        print(f"\n  NOTE: {summary['note']}")
    
    print(f"\n{'='*80}\n")
    
    if test_type.upper() == "SINGLE-LABEL":
        failed = [r for r in results if not (r.get('category_ok') and r.get('flags_ok'))]
    else:
        failed = [r for r in results if not r.get('set_ok')]
    
    if failed:
        print(f"FAILED CASES ({len(failed)} total, showing first 40):")
        print("-" * 80)
        for result in failed[:40]:
            print(f"\n  [{result['id']}] {result['name']}")
            if 'expected_category' in result:
                print(f"    Expected: {result['expected_category']}, Got: {result['got_category']}")
            else:
                print(f"    Expected: {result['expected_categories']}, Got: {result['got_categories']}")
            print(f"    Strategy: {result.get('decision_strategy', 'N/A')}")
    
    print(f"\n{'='*80}\n")


def print_confusion_matrix(confusion: Dict, metrics: Dict):
    categories = sorted(list(confusion.keys()))
    
    print(f"\n{'='*100}")
    print("CONFUSION MATRIX")
    print(f"{'='*100}\n")
    
    col_width = 14
    header = "Actual \\ Pred".ljust(col_width)
    for cat in categories:
        header += cat[:13].ljust(col_width)
    header += "Total".ljust(col_width)
    
    print(header)
    print("-" * (len(header)))
    
    for expected_cat in categories:
        row = expected_cat[:13].ljust(col_width)
        row_total = 0
        for pred_cat in categories:
            count = confusion[expected_cat].get(pred_cat, 0)
            row_total += count
            row += str(count).ljust(col_width)
        row += str(row_total).ljust(col_width)
        print(row)
    
    totals_row = "Total".ljust(col_width)
    for pred_cat in categories:
        col_total = sum(confusion[cat].get(pred_cat, 0) for cat in categories)
        totals_row += str(col_total).ljust(col_width)
    grand_total = sum(confusion[cat][pred] for cat in categories for pred in categories)
    totals_row += str(grand_total).ljust(col_width)
    print("-" * (len(header)))
    print(totals_row)
    
    print(f"\n{'='*100}")
    print("PRECISION / RECALL / F1 SCORES")
    print(f"{'='*100}\n")
    
    metrics_header = "Category".ljust(16) + "Precision".ljust(12) + "Recall".ljust(12) + "F1-Score".ljust(12) + "Support".ljust(10)
    print(metrics_header)
    print("-" * len(metrics_header))
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_support = 0
    
    for cat in categories:
        m = metrics[cat]
        line = cat[:15].ljust(16)
        line += f"{m['precision']:.3f}".ljust(12)
        line += f"{m['recall']:.3f}".ljust(12)
        line += f"{m['f1']:.3f}".ljust(12)
        line += f"{m['support']}".ljust(10)
        print(line)
        
        total_tp += m['tp']
        total_fp += m['fp']
        total_fn += m['fn']
        total_support += m['support']
    
    print("-" * len(metrics_header))
    
    weighted_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    weighted_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0
    
    line = "Weighted Avg".ljust(16)
    line += f"{weighted_precision:.3f}".ljust(12)
    line += f"{weighted_recall:.3f}".ljust(12)
    line += f"{weighted_f1:.3f}".ljust(12)
    line += f"{total_support}".ljust(10)
    print(line)
    
    print(f"\n{'='*100}\n")


def main():
    test_file = Path(__file__).parent / "test_cases.json"
    
    if not test_file.exists():
        print(f"ERROR: Test cases file not found: {test_file}")
        sys.exit(1)
    
    test_cases = load_test_cases(str(test_file))
    print(f"Loaded {len(test_cases)} test cases from {test_file}")
    
    print("\n[1/4] Evaluating single-label classification...")
    single_eval = evaluate_single_label(test_cases, verbose=True)
    print_results(single_eval, "SINGLE-LABEL")
    
    print("[2/4] Evaluating multi-label classification...")
    multi_eval = evaluate_multi_label(test_cases, verbose=True)
    print_results(multi_eval, "MULTI-LABEL")
    
    print("[3/4] Per-category accuracy breakdown...")
    print(f"\n{'='*80}")
    print("PER-CATEGORY ACCURACY")
    print(f"{'='*80}\n")
    category_breakdown = evaluate_by_category(test_cases)
    for cat, stats in sorted(category_breakdown.items()):
        print(f"  {cat:.<25} {stats['correct']:>3}/{stats['total']:<3} "
              f"({stats['accuracy']*100:>5.1f}%)")
    
    print("\n[4/4] Building confusion matrix...")
    confusion, metrics = build_confusion_matrix(test_cases)
    print_confusion_matrix(confusion, metrics)
    
    output_file = Path(__file__).parent / "test_results.json"
    all_results = {
        "single_label": single_eval,
        "multi_label": multi_eval,
        "per_category": category_breakdown,
        "confusion_matrix": {k: {pred: count for pred, count in v.items()} for k, v in confusion.items()},
        "metrics": {cat: metrics[cat] for cat in metrics},
    }
    
    class CategoryEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (Category, RiskFlag)):
                return obj.value
            return super().default(obj)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, cls=CategoryEncoder)
    print(f"Full results saved to {output_file}")


if __name__ == '__main__':
    main()
