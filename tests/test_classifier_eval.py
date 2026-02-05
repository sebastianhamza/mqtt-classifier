import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mqtt_payload_classifier import classify_payload
from mqtt_payload_classifier.constants import Category, RiskFlag

TEST_CASES = [
    {
        "name": "serial_number",
        "payload": '{"registerNr":"SerialNo1", "value":"1115532610"}',
        "expected_category": Category.TELEMETRY,
        "expected_flags": [RiskFlag.LONG_NUMERIC, RiskFlag.SERIAL_NO],
    },
    {
        "name": "ble_gateway",
        "payload": '{"gw_mac":"F9:DE:50:E3:76:6D","rssi":-76,"aoa":[],"gwts":1764517866,"ts":1764517866,"data":"0201181BFF...","coords":""}',
        "expected_category": Category.EVENTS,
        "expected_flags": [RiskFlag.MAC_ADDR, RiskFlag.LONG_NUMERIC],
    },
    {
        "name": "vision_objects_json",
        "payload": '{"objects": [{"confidence": 0.87, "frame": 46264255, "oid": 31218, "type": "person"}], "ts": 1764517866516}',
        "expected_category": Category.AI_INFERENCE,
        "expected_flags": [RiskFlag.LONG_NUMERIC],
    },
    {
        "name": "robotics_position",
        "payload": '{"Bezeichner":"TCP-Kuka", "Position":{"x":277.47,"y":8984.67,"z":842.84}}',
        "expected_category": Category.TELEMETRY,
        "expected_flags": [],
    },
    {
        "name": "free_text_person",
        "payload": 'Person detected near entrance with confidence 0.92',
        "expected_category": Category.AI_INFERENCE,
        "expected_flags": [],
    },
    {
        "name": "battery_voltage",
        "payload": 'battery_voltage: 234.5 V',
        "expected_category": Category.TELEMETRY,
        "expected_flags": [],
    },
    {
        "name": "large_wh",
        "payload": '15286825.6 Wh',
        "expected_category": Category.TELEMETRY,
        "expected_flags": [RiskFlag.LONG_NUMERIC],
    },
    {
        "name": "ble_csv",
        "payload": 'rssi:-55,adv,mac:AA:BB:CC:DD:EE:FF',
        "expected_category": Category.TELEMETRY,
        "expected_flags": [RiskFlag.MAC_ADDR],
    },
    {
        "name": "log_error",
        "payload": '[1556121218] <err> exception occurred in module sensor',
        "expected_category": Category.DEVICE_HEALTH,
        "expected_flags": [RiskFlag.LONG_NUMERIC],
    },
    {
        "name": "ipv_mixture",
        "payload": '192.168.1.194 172.17.0.1 fd7c:3b9f:96d8:ee41:d954:c9dc:e205:7a1e',
        "expected_category": Category.TELEMETRY,
        "expected_flags": [RiskFlag.IPV4_LOCAL, RiskFlag.IPV6],
    },
    {
        "name": "email_and_password",
        "payload": 'user: alice@example.com, action: login, pwd: hunter2, session: 9876543210',
        "expected_category": Category.UNSTRUCTURED,
        "expected_flags": [RiskFlag.EMAIL, RiskFlag.CREDENTIAL, RiskFlag.LONG_NUMERIC],
    },
    {
        "name": "gps_coords",
        "payload": 'Location: 48.8584,2.2945; accuracy: 5m',
        "expected_category": Category.UNSTRUCTURED,
        "expected_flags": [RiskFlag.GPS_PRECISE],
    },
    {
        "name": "long_hex_blob",
        "payload": 'A1B2C3D4E5F60718293A4B5C6D7E8F90123456789ABCDEF0123456789',
        "expected_category": Category.UNSTRUCTURED,
        "expected_flags": [], 
    },
    {
        "name": "serial_and_mac",
        "payload": 'serialno: 998877665544, device_mac: 01:23:45:67:89:AB',
        "expected_category": Category.TELEMETRY,
        "expected_flags": [RiskFlag.SERIAL_NO, RiskFlag.MAC_ADDR, RiskFlag.LONG_NUMERIC],
    },
]


def evaluate():
    results = []
    cat_correct = 0
    flags_correct = 0
    both_correct = 0

    total_flag_score = 0.0
    for case in TEST_CASES:
        r = classify_payload(case["payload"])
        got_cat = r.categories[0]
        got_flags = set(r.risk_flags or [])
        exp_cat = case["expected_category"]
        exp_flags = set(case["expected_flags"])

        ok_cat = (got_cat == exp_cat)
        if len(exp_flags) == 0:
            ok_flags = (len(got_flags) == 0)
            flag_score = 1.0 if ok_flags else 0.0
        else:
            matched = len(exp_flags.intersection(got_flags))
            flag_score = matched / len(exp_flags)
            ok_flags = (matched == len(exp_flags))

        total_flag_score += flag_score

        if ok_cat:
            cat_correct += 1
        if ok_flags:
            flags_correct += 1
        if ok_cat and ok_flags:
            both_correct += 1

        results.append({
            "name": case["name"],
            "payload": (case["payload"][:120] + '...') if len(case["payload"]) > 120 else case["payload"],
            "expected_category": exp_cat,
            "got_category": got_cat,
            "expected_flags": sorted(list(exp_flags)),
            "got_flags": sorted(list(got_flags)),
            "category_ok": ok_cat,
            "flags_ok": ok_flags,
            "flag_score": round(flag_score, 3),
        })

    total = len(TEST_CASES)
    summary = {
        "total": total,
        "category_accuracy": cat_correct / total,
        "flags_accuracy": flags_correct / total,
        "both_accuracy": both_correct / total,
        "avg_flag_match_score": total_flag_score / total,
    }

    print(json.dumps({"cases": results, "summary": summary}, indent=2))


if __name__ == '__main__':
    evaluate()