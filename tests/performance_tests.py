import time
import json
import random
import string
import sys
from typing import List, Tuple, Dict, Any
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mqtt_payload_classifier import classify_payload, classify_many, classify_many_parallel


SAMPLE_PAYLOADS = [
    '{"registerNr":"SerialNo1", "value":"1115532610"}',
    '{"gw_mac":"F9:DE:50:E3:76:6D","rssi":-76,"aoa":[],"gwts":1764517866,"ts":1764517866,"data":"0201181BFF7500420401017668FCCA8A498D6AFCCA8A498C01000000000000","coords":""}',
    '{"objects": [{"confidence": 0.87, "frame": 46264255, "oid": 31218, "type": "person"}], "ts": 1764517866516}',
    '{"Bezeichner":"TCP-Kuka", "Position":{"x":277.47,"y":8984.67,"z":842.84}}',
    'Person detected near entrance with confidence 0.92',
    'battery_voltage: 234.5 V',
    '15286825.6 Wh',
    'rssi:-55,adv,mac:AA:BB:CC:DD:EE:FF',
    '[1556121218] <err> exception occurred in module sensor',
    '192.168.1.194 172.17.0.1 fd7c:3b9f:96d8:ee41:d954:c9dc:e205:7a1e',
    'ON',
    '0.90',
    '3.15 A',
    '{"time":1764517866858,"value":0}',
    '1764525054,649,0',
    'LargeAlarmWireRes =0',
    '{"gw_mac":"D5:C6:9C:D9:22:02","rssi":-67,"aoa":[],"gwts":1764517866,"ts":1764517866,"data":"2BFF9904E10FD356F8AEAF001000140017001802E44200FFFFFFFFFFFF00122BB8FFFFFFFFFFC39ACAB140A2030398FC","coords":""}',
    '245.0',
    '76.5 V',
]


def generate_random_payload(size_bytes: int) -> str:
    base = '{"data":"'
    suffix = '"}'
    overhead = len(base) + len(suffix)
    
    if size_bytes <= overhead:
        return base + suffix
    
    data_size = size_bytes - overhead
    random_data = ''.join(random.choices(string.ascii_letters + string.digits, k=data_size))
    return base + random_data + suffix


def generate_json_payload(size_bytes: int) -> str:
    base_obj = {
        "timestamp": 1764517866,
        "sensor_id": "sensor_001",
        "location": "warehouse_02",
        "readings": []
    }
    
    base_str = json.dumps(base_obj)
    overhead = len(base_str)
    
    if size_bytes <= overhead:
        return base_str
    
    remaining = size_bytes - overhead
    reading_size = 50
    num_readings = max(1, remaining // reading_size)
    
    for i in range(num_readings):
        base_obj["readings"].append({
            "id": i,
            "value": random.random() * 100,
            "unit": "units",
            "status": random.choice(["OK", "WARNING", "ERROR"])
        })
    
    return json.dumps(base_obj)


def benchmark_single_payload(payload: str, iterations: int = 1000) -> Dict[str, float]:
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        classify_payload(payload)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    times.sort()
    return {
        "min_ms": min(times),
        "max_ms": max(times),
        "mean_ms": sum(times) / len(times),
        "median_ms": times[len(times) // 2],
        "total_iterations": iterations,
    }


def load_test(num_packets: int, use_samples: bool = True) -> Dict[str, Any]:
    payloads = []
    
    if use_samples:
        for i in range(num_packets):
            payloads.append(SAMPLE_PAYLOADS[i % len(SAMPLE_PAYLOADS)])
    else:
        for _ in range(num_packets):
            payloads.append(generate_random_payload(random.randint(50, 500)))
    
    #Serial
    start = time.perf_counter()
    results_serial = classify_many(payloads)
    end = time.perf_counter()
    
    elapsed_serial = end - start
    throughput_serial = num_packets / elapsed_serial
    
    #Parallel
    start = time.perf_counter()
    results_parallel = classify_many_parallel(payloads)
    end = time.perf_counter()
    
    elapsed_parallel = end - start
    throughput_parallel = num_packets / elapsed_parallel
    
    return {
        "num_packets": num_packets,
        "serial": {
            "elapsed_seconds": elapsed_serial,
            "packets_per_second": throughput_serial,
            "ms_per_packet": (elapsed_serial / num_packets) * 1000,
            "processed_results": len(results_serial),
        },
        "parallel": {
            "elapsed_seconds": elapsed_parallel,
            "packets_per_second": throughput_parallel,
            "ms_per_packet": (elapsed_parallel / num_packets) * 1000,
            "processed_results": len(results_parallel),
        },
        "speedup": elapsed_serial / elapsed_parallel,
    }


def payload_size_impact(max_size_bytes: int = 10000, step_size: int = 1000) -> List[Dict[str, Any]]:
    results = []
    sizes = range(100, max_size_bytes + 1, step_size)
    
    for size in sizes:
        payload = generate_json_payload(size)
        actual_size = len(payload.encode('utf-8'))
        
        timing = benchmark_single_payload(payload, iterations=100)
        
        results.append({
            "requested_size_bytes": size,
            "actual_size_bytes": actual_size,
            "mean_ms": timing["mean_ms"],
            "median_ms": timing["median_ms"],
            "min_ms": timing["min_ms"],
            "max_ms": timing["max_ms"],
        })
    
    return results


def concurrent_batch_processing(batch_sizes: List[int] = None) -> List[Dict[str, Any]]:
    if batch_sizes is None:
        batch_sizes = [10, 50, 100, 500, 1000, 5000]
    
    results = []
    
    for batch_size in batch_sizes:
        payloads = [SAMPLE_PAYLOADS[i % len(SAMPLE_PAYLOADS)] for i in range(batch_size)]
        
        #Serial
        start = time.perf_counter()
        batch_results_serial = classify_many(payloads)
        end = time.perf_counter()
        
        elapsed_serial = end - start
        throughput_serial = batch_size / elapsed_serial
        
        #Parallel
        start = time.perf_counter()
        batch_results_parallel = classify_many_parallel(payloads)
        end = time.perf_counter()
        
        elapsed_parallel = end - start
        throughput_parallel = batch_size / elapsed_parallel
        
        results.append({
            "batch_size": batch_size,
            "serial": {
                "elapsed_seconds": elapsed_serial,
                "packets_per_second": throughput_serial,
                "ms_per_packet": (elapsed_serial / batch_size) * 1000,
            },
            "parallel": {
                "elapsed_seconds": elapsed_parallel,
                "packets_per_second": throughput_parallel,
                "ms_per_packet": (elapsed_parallel / batch_size) * 1000,
            },
            "speedup": elapsed_serial / elapsed_parallel,
        })
    
    return results


def memory_usage_test(payload_count: int = 10000) -> Dict[str, Any]:
    payloads = [SAMPLE_PAYLOADS[i % len(SAMPLE_PAYLOADS)] for i in range(payload_count)]
    
    total_payload_size = sum(len(p.encode('utf-8')) for p in payloads) / (1024 * 1024)
    
    #Serial
    start = time.perf_counter()
    results_serial = classify_many(payloads)
    end = time.perf_counter()
    elapsed_serial = end - start
    
    #Parallel
    start = time.perf_counter()
    results_parallel = classify_many_parallel(payloads)
    end = time.perf_counter()
    elapsed_parallel = end - start
    
    estimated_result_size = (len(results_serial) * 500) / (1024 * 1024)
    
    return {
        "payload_count": payload_count,
        "total_payload_size_mb": round(total_payload_size, 2),
        "estimated_result_size_mb": round(estimated_result_size, 2),
        "estimated_total_memory_mb": round(total_payload_size + estimated_result_size, 2),
        "serial_processing_time_seconds": round(elapsed_serial, 2),
        "parallel_processing_time_seconds": round(elapsed_parallel, 2),
        "speedup": round(elapsed_serial / elapsed_parallel, 2),
    }


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_results(data: Any, indent: int = 0) -> None:
    indent_str = " " * indent
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{indent_str}{key}:")
                print_results(value, indent + 2)
            elif isinstance(value, float):
                print(f"{indent_str}{key}: {value:.4f}")
            else:
                print(f"{indent_str}{key}: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print(f"{indent_str}[{i}]")
            print_results(item, indent + 2)


def main():
    print_section("1. LOAD TESTING (Serial vs Parallel)")
    print("Testing throughput with different packet counts...\n")
    
    load_test_sizes = [100, 500, 1000, 5000]
    load_test_results = []
    
    for size in load_test_sizes:
        print(f"Processing {size} packets (using sample payloads)...")
        result = load_test(size, use_samples=True)
        load_test_results.append(result)
        print(f"  Serial:    {result['serial']['packets_per_second']:.2f} packets/sec ({result['serial']['ms_per_packet']:.4f} ms/packet)")
        print(f"  Paralel:  {result['parallel']['packets_per_second']:.2f} packets/sec ({result['parallel']['ms_per_packet']:.4f} ms/packet)")
        print(f"  Speedup:  {result['speedup']:.2f}x\n")
    

    print_section("2. PAYLOAD SIZE IMPACT")
    print("Testing classification speed across different payload sizes\n")
    
    size_results = payload_size_impact(max_size_bytes=5000, step_size=500)
    
    print(f"{'Size (bytes)':<15} {'Mean (ms)':<12} {'Median (ms)':<12} {'Min (ms)':<10} {'Max (ms)':<10}")
    print("-" * 59)
    for result in size_results:
        print(f"{result['actual_size_bytes']:<15} {result['mean_ms']:<12.4f} {result['median_ms']:<12.4f} {result['min_ms']:<10.4f} {result['max_ms']:<10.4f}")
    

    print_section("3. BATCH PROCESSING EFFICIENCY (Serial vs Paralel)")
    print("Testing different batch sizes to measure efficiency...\n")
    
    batch_results = concurrent_batch_processing()
    
    print(f"{'Batch Size':<12} {'Mode':<10} {'Packets/Sec':<18} {'Ms/Packet':<12}")
    print("-" * 52)
    for result in batch_results:
        print(f"{result['batch_size']:<12} {'Serial':<10} {result['serial']['packets_per_second']:<18.2f} {result['serial']['ms_per_packet']:<12.4f}")
        print(f"{'':<12} {'Parallel':<10} {result['parallel']['packets_per_second']:<18.2f} {result['parallel']['ms_per_packet']:<12.4f}")
        print(f"{'':<12} {'Speedup':<10} {result['speedup']:<18.2f}x")
        print()
    

    print_section("4. MEMORY USAGE ESTIMATION (Serial vs Parallel)")
    print("Estimating memory usage for large-scale processing...\n")
    
    memory_result = memory_usage_test(payload_count=5000)
    print_results(memory_result)
    

    print_section("5. INDIVIDUAL PAYLOAD LATENCY")
    print("Analyzing latency for individual payloads...\n")
    
    sample_timing = benchmark_single_payload(SAMPLE_PAYLOADS[0], iterations=1000)
    print_results(sample_timing)
    

    print_section("PERFORMANCE SUMMARY")
    
    best_load_serial = max(load_test_results, key=lambda x: x['serial']['packets_per_second'])
    best_load_parallel = max(load_test_results, key=lambda x: x['parallel']['packets_per_second'])
    print(f"Best serial throughput: {best_load_serial['serial']['packets_per_second']:.2f} packets/sec")
    print(f"  (with {best_load_serial['num_packets']} packets, {best_load_serial['serial']['elapsed_seconds']:.4f}s)")
    print(f"\nBest parallel throughput: {best_load_parallel['parallel']['packets_per_second']:.2f} packets/sec")
    print(f"  (with {best_load_parallel['num_packets']} packets, {best_load_parallel['parallel']['elapsed_seconds']:.4f}s)")
    
    avg_speedup = sum(r['speedup'] for r in load_test_results) / len(load_test_results)
    print(f"\nAverage speedup (parallel vs serial): {avg_speedup:.2f}x")
    
    size_impact = size_results[-1]['mean_ms'] / size_results[0]['mean_ms']
    print(f"Payload size impact: {size_impact:.2f}x slower (max vs min size)")
    
    best_batch_parallel = max(batch_results, key=lambda x: x['parallel']['packets_per_second'])
    print(f"\nOptimal batch size (parallel): {best_batch_parallel['batch_size']} packets")
    print(f"  Throughput: {best_batch_parallel['parallel']['packets_per_second']:.2f} packets/sec")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
