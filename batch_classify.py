from mqtt_payload_classifier.batch_processor import process_payloads


def main():
    input_file = "resources/input_full.txt"
    output_file = "resources/output.txt"

    process_payloads(input_file, output_file, format_type='jsonl', sanitize=False, use_parallel=True)


if __name__ == '__main__':
    main()
