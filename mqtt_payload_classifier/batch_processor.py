import json
import time
from pathlib import Path
from typing import List

from .classifier import classify_payload, classify_many_parallel


def load_payloads(input_file: str) -> List[str]:
    payloads = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    
    lines = content.split('\n')
    current_payload = []
    in_json = False
    bracket_count = 0
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            if current_payload:
                payloads.append('\n'.join(current_payload))
                current_payload = []
                in_json = False
                bracket_count = 0
        else:
            if stripped[0] in ('{', '['):
                if current_payload and not in_json:
                    payloads.append('\n'.join(current_payload))
                    current_payload = []
                
                in_json = True
                current_payload = [line]
                bracket_count = stripped.count('{') + stripped.count('[') - \
                                stripped.count('}') - stripped.count(']')
            else:
                if in_json:
                    current_payload.append(line)
                    bracket_count += stripped.count('{') + stripped.count('[') - \
                                    stripped.count('}') - stripped.count(']')
                    
                    if bracket_count == 0:
                        payloads.append('\n'.join(current_payload))
                        current_payload = []
                        in_json = False
                else:
                    if current_payload and not stripped[0].isspace():
                        payloads.append('\n'.join(current_payload))
                        current_payload = [line]
                    else:
                        current_payload.append(line)
    
    if current_payload:
        payloads.append('\n'.join(current_payload))
    
    return payloads


def classify_batch(payloads: List[str], sanitize: bool = True, show_progress: bool = True) -> List[dict]:
    results = []
    
    for i, payload in enumerate(payloads):
        result = classify_payload(payload, sanitize=sanitize)
        results.append({
            'payload': payload,
            'structure': result.structure,
            'categories': [str(cat) for cat in result.categories],
            'category_scores': {str(k): v for k, v in result.category_scores.items()},
            'risk_flags': [str(f) for f in result.risk_flags],
            'risk_score': result.risk_score,
            'decision_strategy': result.decision_strategy,
            'heuristic_categories': {str(k): v for k, v in result.heuristic_categories.items()},
            'semantic_categories': {str(k): v for k, v in result.semantic_categories.items()},
        })
        
        if show_progress and (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(payloads)} payloads ({((i + 1) / len(payloads) * 100):.1f}%)", flush=True)
    
    if show_progress and len(payloads) % 500 != 0:
        print(f"  Processed {len(payloads)}/{len(payloads)} payloads (100.0%)", flush=True)
    
    return results


def save_results(results: List[dict], output_file: str, format_type: str = 'json') -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if format_type == 'jsonl':
            for result in results:
                f.write(json.dumps(result) + '\n')
        else:
            json.dump(results, f, indent=2, ensure_ascii=False)


def process_payloads(input_file: str, output_file: str, format_type: str = 'json', sanitize: bool = True, use_parallel: bool = False, num_workers: int = None) -> None:
    start_time = time.time()
    
    print(f"Loading payloads from {input_file}...")
    payloads = load_payloads(input_file)
    print(f"Loaded {len(payloads)} payloads")
    
    print(f"Classifying {len(payloads)} payloads{'  (using parallel processing)' if use_parallel else ''}...")
    
    classify_start = time.time()
    if use_parallel and len(payloads) > 100:

        print(f"Classifying in parallel using {num_workers or 'default'} workers...")
        results_parallel = classify_many_parallel(payloads, sanitize=sanitize, num_workers=num_workers, show_progress=True)
        results = []
        for result in results_parallel:
            results.append({
                'payload': result.payload,
                'structure': result.structure,
                'categories': [str(cat) for cat in result.categories],
                'category_scores': {str(k): v for k, v in result.category_scores.items()},
                'risk_flags': [str(f) for f in result.risk_flags],
                'risk_score': result.risk_score,
                'decision_strategy': result.decision_strategy,
                'heuristic_categories': {str(k): v for k, v in result.heuristic_categories.items()},
                'semantic_categories': {str(k): v for k, v in result.semantic_categories.items()},
            })
    else:
        print(f"Classifying serial")
        results = classify_batch(payloads, sanitize=sanitize, show_progress=True)
    
    classify_end = time.time()
    
    save_results(results, output_file, format_type)
    end_time = time.time()
    
    total_time = end_time - start_time
    classify_time = classify_end - classify_start
    
    print(f"Results saved to {output_file}")
    print(f"\nProcessing Summary:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Classification time: {classify_time:.2f}s")
    print(f"  Throughput: {len(payloads) / classify_time:.2f} payloads/sec")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python batch_processor.py <input_file> <output_file> [format_type] [--no-sanitize] [--parallel]")
        print("       format_type: 'json' (default) or 'jsonl'")
        print("       --no-sanitize: disable sample sanitization (keep full numbers)")
        print("       --parallel: use multiprocessing for faster batch processing (10k+ payloads)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    format_type = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 'json'
    sanitize = '--no-sanitize' not in sys.argv
    use_parallel = '--parallel' in sys.argv
    
    process_payloads(input_file, output_file, format_type, sanitize=sanitize, use_parallel=use_parallel)
