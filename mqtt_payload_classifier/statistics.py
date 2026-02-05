"""Stats and reporting for classified MQTT payloads."""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter, defaultdict


class ResultAnalyzer:
    """Analyzes classification results and builds stats reports."""
    
    def __init__(self, results: List[dict], deduplicate: bool = True):
        """Set up analyzer. Optionally deduplicates payloads before analysis."""
        self.total_loaded = len(results)
        self.deduplicate = deduplicate
        
        if deduplicate:
            self.results, self.duplicates_skipped = self._deduplicate_results(results)
        else:
            self.results = results
            self.duplicates_skipped = 0
        
        self.total_count = len(self.results)
    
    def _deduplicate_results(self, results: List[dict]) -> tuple:
        """Remove duplicate payloads, keep first occurrence. Returns (unique, skipped_count)."""
        seen_payloads = set()
        unique_results = []
        duplicates_skipped = 0
        
        for result in results:
            payload = result['payload']
            if payload not in seen_payloads:
                seen_payloads.add(payload)
                unique_results.append(result)
            else:
                duplicates_skipped += 1
        
        return unique_results, duplicates_skipped
    
    def get_deduplication_stats(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        return {
            'total_payloads_loaded': self.total_loaded,
            'payloads_skipped_duplicates': self.duplicates_skipped,
            'total_payloads_analyzed': self.total_count
        }
    
    def get_structure_distribution(self) -> Dict[str, int]:
        """Get count of payloads by structure type."""
        structures = Counter(r['structure'] for r in self.results)
        return dict(structures)
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get count of payloads by each category (multi-category support)."""
        category_counter = Counter()
        for result in self.results:
            for category in result['categories']:
                category_counter[category] += 1
        return dict(category_counter)
    
    def get_primary_category_distribution(self) -> Dict[str, int]:
        """Get count of payloads by their primary (first) category only."""
        primary_categories = Counter()
        for result in self.results:
            if result['categories']:
                primary_category = result['categories'][0]
                primary_categories[primary_category] += 1
        return dict(primary_categories)
    
    def get_confidence_stats(self) -> Dict[str, Any]:
        """Get statistics about category scores (confidence)."""
        all_scores = []
        for result in self.results:
            all_scores.extend(result['category_scores'].values())
        
        if not all_scores:
            return {'min': 0, 'max': 0, 'avg': 0, 'median': 0}
        
        sorted_scores = sorted(all_scores)
        return {
            'min': min(sorted_scores),
            'max': max(sorted_scores),
            'avg': sum(sorted_scores) / len(sorted_scores),
            'median': sorted_scores[len(sorted_scores) // 2]
        }
    
    def get_semantic_coverage(self) -> Dict[str, Any]:
        """Get statistics about semantic classifier usage via decision_strategy."""
        strategy_counter = Counter(r['decision_strategy'] for r in self.results)
        
        heuristic_only = strategy_counter.get('heuristic', 0)
        semantic_only = strategy_counter.get('semantic', 0)
        hybrid = strategy_counter.get('hybrid', 0)
        used_semantic = semantic_only + hybrid
        
        return {
            'heuristic_only': heuristic_only,
            'semantic_only': semantic_only,
            'hybrid': hybrid,
            'none': strategy_counter.get('none', 0),
            'total_with_semantic': used_semantic,
            'percentage_with_semantic': round((used_semantic / self.total_count * 100), 2) if self.total_count > 0 else 0,
            'strategy_distribution': dict(strategy_counter)
        }
    
    def get_risk_analysis(self) -> Dict[str, Any]:
        """Analyze risk flags and scores."""
        risk_flags_counter = Counter()
        risk_scores = [r['risk_score'] for r in self.results]
        high_risk_count = sum(1 for score in risk_scores if score > 5)
        
        # Count payloads with at least one risk flag
        payloads_with_flags = sum(1 for result in self.results if result['risk_flags'])
        
        for result in self.results:
            for flag in result['risk_flags']:
                # Clean up RiskFlag. prefix if present
                flag_name = flag.replace('RiskFlag.', '')
                risk_flags_counter[flag_name] += 1
        
        return {
            'avg_risk_score': round(sum(risk_scores) / len(risk_scores), 2),
            'max_risk_score': max(risk_scores),
            'min_risk_score': min(risk_scores),
            'high_risk_count': high_risk_count,
            'high_risk_percentage': round((high_risk_count / self.total_count * 100), 2),
            'payloads_with_flags_count': payloads_with_flags,
            'payloads_with_flags_percentage': round((payloads_with_flags / self.total_count * 100), 2),
            'most_common_flags': dict(risk_flags_counter.most_common(5)),
            'all_flags': dict(risk_flags_counter)
        }
    
    def get_classifier_usage(self) -> Dict[str, int]:
        """Get count of results by decision strategy (heuristic vs semantic)."""
        usage = Counter(r['decision_strategy'] for r in self.results)
        return dict(usage)
    
    def get_confidence_by_category(self) -> Dict[str, Dict[str, Any]]:
        """Get average score for each category (multi-category support)."""
        category_scores = defaultdict(list)
        
        for result in self.results:
            # Handle category_scores dict from new format
            for category, score in result['category_scores'].items():
                category_scores[category].append(score)
        
        return {
            cat: {
                'avg_score': round(sum(scores) / len(scores), 3),
                'count': len(scores),
                'min_score': round(min(scores), 3),
                'max_score': round(max(scores), 3)
            }
            for cat, scores in category_scores.items()
        }
    
    def get_structure_category_matrix(self) -> Dict[str, Dict[str, int]]:
        """Get distribution of categories within each structure type (multi-category support)."""
        matrix = defaultdict(lambda: defaultdict(int))
        
        for result in self.results:
            struct = result['structure']
            for category in result['categories']:
                matrix[struct][category] += 1
        
        return {struct: dict(cats) for struct, cats in matrix.items()}
    
    def get_multi_category_analysis(self) -> Dict[str, Any]:
        """Analyze payloads with multiple categories."""
        single_cat_count = 0
        multi_cat_count = 0
        multi_cat_combinations = defaultdict(int)
        multi_cat_category_pairs = defaultdict(int)
        
        for result in self.results:
            categories = result['categories']
            num_cats = len(categories)
            
            if num_cats <= 1:
                single_cat_count += 1
            else:
                multi_cat_count += 1
                # Sort to create consistent combination key
                combo_key = ", ".join(sorted(categories))
                multi_cat_combinations[combo_key] += 1
                
                # Track pairs of categories
                for i, cat1 in enumerate(categories):
                    for cat2 in categories[i+1:]:
                        pair = tuple(sorted([cat1, cat2]))
                        multi_cat_category_pairs[pair] += 1
        
        return {
            'single_category_count': single_cat_count,
            'multi_category_count': multi_cat_count,
            'multi_category_percentage': round((multi_cat_count / self.total_count * 100), 2) if self.total_count > 0 else 0,
            'top_combinations': dict(sorted(multi_cat_combinations.items(), key=lambda x: x[1], reverse=True)[:10]),
            'category_pair_frequency': {f"{c1} + {c2}": count for (c1, c2), count in sorted(multi_cat_category_pairs.items(), key=lambda x: x[1], reverse=True)[:10]}
        }
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Build the full summary stats report."""
        return {
            **self.get_deduplication_stats(),
            'structure_distribution': self.get_structure_distribution(),
            'category_distribution': self.get_category_distribution(),
            'primary_category_distribution': self.get_primary_category_distribution(),
            'multi_category_analysis': self.get_multi_category_analysis(),
            'confidence_statistics': self.get_confidence_stats(),
            'semantic_coverage': self.get_semantic_coverage(),
            'classifier_usage': self.get_classifier_usage(),
            'risk_analysis': self.get_risk_analysis(),
            'confidence_by_category': self.get_confidence_by_category(),
            'structure_category_matrix': self.get_structure_category_matrix()
        }


def load_results(results_file: str) -> List[dict]:
    """Load classification results from a JSONL file."""
    results = []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    
    return results


def save_statistics(stats: Dict[str, Any], output_file: str) -> None:
    """Write stats dict to a JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def print_statistics(stats: Dict[str, Any]) -> None:
    """Print formatted stats to stdout."""
    print("\n" + "=" * 70)
    print("MQTT PAYLOAD CLASSIFICATION STATISTICS")
    print("=" * 70)
    
    print(f"\nSUMMARY")
    print(f"  Total Payloads Loaded: {stats['total_payloads_loaded']}")
    if stats['payloads_skipped_duplicates'] > 0:
        dup_pct = round((stats['payloads_skipped_duplicates'] / stats['total_payloads_loaded'] * 100), 2)
        print(f"  Duplicates Skipped: {stats['payloads_skipped_duplicates']} ({dup_pct}%)")
    print(f"  Payloads Analyzed: {stats['total_payloads_analyzed']}")
    
    print(f"\nSTRUCTURE DISTRIBUTION")
    for struct, count in sorted(stats['structure_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = round((count / stats['total_payloads_analyzed'] * 100), 2)
        print(f"  {struct}: {count} ({percentage}%)")
    
    print(f"\nPRIMARY CATEGORY DISTRIBUTION")
    for cat, count in sorted(stats['primary_category_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = round((count / stats['total_payloads_analyzed'] * 100), 2)
        print(f"  {cat}: {count} ({percentage}%)")
    
    print(f"\nCATEGORY DISTRIBUTION (all categories)")
    for cat, count in sorted(stats['category_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = round((count / stats['total_payloads_analyzed'] * 100), 2)
        print(f"  {cat}: {count} ({percentage}%)")
    
    multi = stats['multi_category_analysis']
    print(f"\nMULTI-CATEGORY ANALYSIS")
    print(f"  Single Category: {multi['single_category_count']}")
    print(f"  Multi-Category: {multi['multi_category_count']} ({multi['multi_category_percentage']}%)")
    
    if multi['top_combinations']:
        print(f"\n  Top Combinations:")
        for combo, count in list(multi['top_combinations'].items())[:5]:
            print(f"    - {combo}: {count}")
    
    if multi['category_pair_frequency']:
        print(f"\n  Common Pairs:")
        for pair, count in list(multi['category_pair_frequency'].items())[:5]:
            print(f"    - {pair}: {count}")
    
    print(f"\nSCORE STATISTICS")
    conf = stats['confidence_statistics']
    print(f"  Average: {conf['avg']:.3f}")
    print(f"  Median: {conf['median']:.3f}")
    print(f"  Range: {conf['min']:.3f} - {conf['max']:.3f}")
    
    print(f"\nDECISION STRATEGY")
    for strategy, count in sorted(stats['classifier_usage'].items(), key=lambda x: x[1], reverse=True):
        percentage = round((count / stats['total_payloads_analyzed'] * 100), 2)
        print(f"  {strategy}: {count} ({percentage}%)")
    
    sem = stats['semantic_coverage']
    print(f"\nSEMANTIC USAGE")
    print(f"  Heuristic Only: {sem['heuristic_only']}")
    print(f"  Semantic Only: {sem['semantic_only']}")
    print(f"  Hybrid (Both): {sem['hybrid']}")
    print(f"  No Classification: {sem['none']}")
    print(f"  Total Using Semantic: {sem['total_with_semantic']} ({sem['percentage_with_semantic']}%)")
    
    risk = stats['risk_analysis']
    print(f"\nRISK ANALYSIS")
    print(f"  Avg Risk Score: {risk['avg_risk_score']}")
    print(f"  Risk Score Range: {risk['min_risk_score']} - {risk['max_risk_score']}")
    print(f"  High Risk (>5): {risk['high_risk_count']} ({risk['high_risk_percentage']}%)")
    print(f"  With Risk Flags: {risk['payloads_with_flags_count']} ({risk['payloads_with_flags_percentage']}%)")
    print(f"\n  Risk Flags:")
    for flag, count in sorted(risk['all_flags'].items(), key=lambda x: x[1], reverse=True):
        percentage = round((count / sum(risk['all_flags'].values()) * 100), 2) if risk['all_flags'] else 0
        print(f"    - {flag}: {count} ({percentage}%)")
    
    print(f"\nSCORE BY CATEGORY")
    for cat, stats_dict in sorted(stats['confidence_by_category'].items()):
        print(f"  {cat}:")
        print(f"    Average Score: {stats_dict['avg_score']:.3f}")
        print(f"    Score Range: {stats_dict['min_score']:.3f} - {stats_dict['max_score']:.3f}")
        print(f"    Appearances: {stats_dict['count']}")
    
    print("\n" + "=" * 70 + "\n")


def analyze_results(results_file: str, output_stats_file: str = None, print_output: bool = True, deduplicate: bool = True) -> Dict[str, Any]:
    """Load results, run analysis, and optionally save/print stats."""
    print(f"Loading results from {results_file}...")
    results = load_results(results_file)
    print(f"Loaded {len(results)} results")
    
    print("Generating statistics...")
    analyzer = ResultAnalyzer(results, deduplicate=deduplicate)
    stats = analyzer.generate_summary_report()
    
    if print_output:
        print_statistics(stats)
    
    if output_stats_file:
        print(f"Saving statistics to {output_stats_file}...")
        save_statistics(stats, output_stats_file)
    
    return stats


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python statistics.py <results_file> [output_stats_file] [--no-deduplicate]")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_stats_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    deduplicate = '--no-deduplicate' not in sys.argv
    
    analyze_results(results_file, output_stats_file, print_output=True, deduplicate=deduplicate)
