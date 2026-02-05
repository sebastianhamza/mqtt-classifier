from mqtt_payload_classifier.statistics import analyze_results


def main():
    results_file = "resources/output.txt"
    output_stats_file = "resources/statistics.json"
    deduplicate = True
    
    try:
        analyze_results(
            results_file,
            output_stats_file=output_stats_file,
            print_output=True,
            deduplicate=deduplicate
        )
        print("Analysis complete!")
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
