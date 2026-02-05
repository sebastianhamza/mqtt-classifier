import json
from mqtt_payload_classifier import classify_many

sample_payloads = [
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


def main():
    results = classify_many(sample_payloads)
    for r in results:
        print(json.dumps({
            'structure': r.structure,
            'categories': [cat.value for cat in r.categories],
            'category_scores': {k.value: v for k, v in r.category_scores.items()},
            'decision_strategy': r.decision_strategy,
            'heuristic_categories': [cat.value for cat in r.heuristic_categories],
            'semantic_categories': [cat.value for cat in r.semantic_categories],
            'risk_flags': [flag.value for flag in r.risk_flags],
            'risk_score': r.risk_score,
            'sample': r.payload,
            'decision_used': r.decision_strategy,
        }, indent=2))


if __name__ == '__main__':
    main()
