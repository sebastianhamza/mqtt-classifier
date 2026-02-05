"""Heuristic detection and classification."""

import re
import json
from typing import List, Dict, Set

from .constants import Category, RiskFlag, RISK_WEIGHTS, CATEGORY_KEYWORDS


def _count_keyword_matches(text: str, keywords: Set[str]) -> int:
    count = 0
    for k in keywords:
        if re.search(r'\b' + re.escape(k) + r'\b', text):
            count += 1
    return count


def _get_keyword_specificity(category: Category) -> float:
    specificity_map = {
        Category.AI_INFERENCE: 1.15,
        Category.DEVICE_HEALTH: 1.10,
        Category.EVENTS: 1.05,
        Category.CONTROL: 1.0,
        Category.TELEMETRY: 1.0,
    }
    return specificity_map.get(category, 1.0)


def _looks_like_json(text: str) -> bool:
    t = text.strip()
    
    if not (t.startswith("{") or t.startswith("[")):
        return False
    
    if (t.startswith("{") and t.endswith("}")) or \
       (t.startswith("[") and t.endswith("]")):
        balance = 0
        for c in t:
            if c in "{[":
                balance += 1
            elif c in "}]":
                balance -= 1
        if balance == 0:
            return True
    
    #bracket count
    balance = 0
    for c in t:
        if c in "{[":
            balance += 1
        elif c in "}]":
            balance -= 1
    
    if abs(balance) <= 2 and (balance > 0 or balance <= -1):
        if ":" in t and "," in t:
            return True
        if ":" in t and t.count(":") >= t.count(","):
            return True
    
    return False


def _looks_like_keyvalue(text: str) -> bool:
    kv_pattern = r'\w+[:=][^,\s]+'
    matches = re.findall(kv_pattern, text)
    
    if len(matches) >= 2:
        match_ratio = len(matches) / (len(text.split(',')) + 1)
        return match_ratio > 0.5
    
    return False


def _looks_like_log(text: str) -> bool:
    t = text.lower()
    
    if re.search(r'\[\d{9,}\]', text):
        return True
    
    log_keywords = {"warn", "error", "fail", "exception", "critical", "log", "[err]", "[warn]"}
    if any(kw in t for kw in log_keywords):
        return True
    
    if "traceback" in t or "at line" in t:
        return True
    
    return False


def _looks_like_csv_improved(text: str) -> bool:
    if text.startswith("{") or text.startswith("["):
        return False
    
    parts = text.split(",")
    
    if len(parts) < 3:
        return False
    
    json_like_parts = 0
    for p in parts:
        p_stripped = p.strip()
        if ":" in p_stripped and not "://" in p_stripped:
            json_like_parts += 1
    
    if len(parts) > 0 and (json_like_parts / len(parts)) > 0.3:
        return False
    
    valid_parts = 0
    for p in parts:
        p_stripped = p.strip()
        if len(p_stripped) > 0 and (re.search(r'\w', p_stripped)):
            valid_parts += 1
    
    return (valid_parts / len(parts)) >= 0.7


def _detect_ble_event(data: dict, keys_lower: set) -> float:
    ble_indicators = {
        "gw_mac": 0.4,
        "gateway_mac": 0.4,
        "ibeacon": 0.4,
        "ble": 0.4,
        "beacon": 0.4,
        "advertisement": 0.35,
        "adv": 0.30,
        "aoa": 0.25,
        "rssi": 0.15,
        "mac": 0.10,
        "gwts": 0.15,
        "ts": 0.05,
    }
    
    found_indicators = []
    for key in keys_lower:
        normalized = key.replace("_", "").replace("-", "").lower()
        
        for indicator, weight in ble_indicators.items():
            indicator_norm = indicator.replace("_", "").replace("-", "").lower()
            
            if indicator_norm == normalized:
                found_indicators.append(weight)
                break
    
    if not found_indicators:
        return 0.0
    
    strong_indicators = [w for w in found_indicators if w > 0.3]
    medium_indicators = [w for w in found_indicators if 0.2 <= w <= 0.3]
    
    if not strong_indicators and len(medium_indicators) < 2:
        return 0.0
    
    confidence = min(0.95, sum(found_indicators))
    
    if len(strong_indicators) >= 2:
        confidence = min(0.95, confidence + 0.1)
    
    return round(confidence, 3)

# Regex patterns
HEX_STR_RE = re.compile(r"^[0-9A-Fa-f]+$")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6_RE = re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b")
MAC_RE = re.compile(r"([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}")
GPS_RE = re.compile(r"[-+]?\d{1,3}\.\d{4,},\s*[-+]?\d{1,3}\.\d{4,}")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
TIMESTAMP_LOG_RE = re.compile(r"\[\d{9,}\]")


KEYWORDS_MAP: Dict[Category, Set[str]] = {
    cat: set(k.lower() for k in terms)
    for cat, terms in CATEGORY_KEYWORDS.items()
}

AMBIGUOUS_BLE_KEYWORDS = {"rssi", "mac", "beacon", "adv"}
EVENTS_ONLY_KEYWORDS = {"advertisement", "ble", "event", "motion", "trigger", "update", "notification", "change"}

def is_json_object(s: str) -> bool:
    t = s.strip()
    return len(t) > 2 and t.startswith("{") and t.endswith("}")


def is_json_array(s: str) -> bool:
    t = s.strip()
    return len(t) > 2 and t.startswith("[") and t.endswith("]")


def is_hex_blob(s: str) -> bool:
    s2 = s.strip().replace(" ", "").replace("\n", "")
    return len(s2) >= 32 and HEX_STR_RE.match(s2) is not None


def is_single_numeric(s: str) -> bool:
    t = s.strip()
    return re.fullmatch(r"[-+]?\d+(\.\d+)?", t) is not None


def is_ip_address(s: str) -> bool:
    #Detect if payload is a IP address
    t = s.strip()
    ipv4_match = IPV4_RE.findall(t)
    if len(ipv4_match) == 1 and len(t) <= 20:
        return True
    ipv6_match = IPV6_RE.findall(t)
    if len(ipv6_match) == 1 and len(t) <= 50:
        return True
    return False


def is_base64_or_encrypted(s: str) -> bool:
    #Detect base64 or encrypted-looking data
    t = s.strip()
    if len(t) < 16:
        return False
    
    base64_pattern = r'^[A-Za-z0-9+/]{12,}={0,2}$'
    if re.match(base64_pattern, t):
        return True
    
    if len(t) > 32 and re.match(r'^[0-9A-Fa-f]+$', t):
        return True
    
    return False



def looks_like_csv(s: str) -> bool:
    parts = s.split(",")
    if len(parts) < 3:
        return False
    return any(re.search(r'\d', p) for p in parts)


def contains_whole_word(text: str, keywords: Set[str]) -> bool:
    for k in keywords:
        if re.search(r'\b' + re.escape(k) + r'\b', text):
            return True
    return False


def detect_structure(payload: str) -> str:
    p = payload.strip()
    
    if is_json_object(p):
        return "json_object"
    if is_json_array(p):
        return "json_array"
    
    if _looks_like_json(p):
        return "json_like"
    
    if is_hex_blob(p):
        return "hex_blob"
    if is_single_numeric(p):
        return "numeric"
    if _looks_like_log(p):
        return "log_line"
    if _looks_like_keyvalue(p):
        return "keyvalue"
    if _looks_like_csv_improved(p):
        return "csv_like"
    
    return "text"


def detect_sensitive(payload: str) -> List[RiskFlag]:
    flags: List[RiskFlag] = []
    
    MAX_SENSITIVE_CHECK = 5000
    check_payload = payload[:MAX_SENSITIVE_CHECK]
    
    if EMAIL_RE.search(check_payload):
        flags.append(RiskFlag.EMAIL)
    
    m4 = IPV4_RE.search(check_payload)
    if m4:
        ip = m4.group(0)
        if ip.startswith("10.") or ip.startswith("192.168.") or ip.startswith("172."):
            flags.append(RiskFlag.IPV4_LOCAL)
        else:
            flags.append(RiskFlag.IPV4_PUBLIC)
            
    if IPV6_RE.search(check_payload):
        flags.append(RiskFlag.IPV6)
    if MAC_RE.search(check_payload):
        flags.append(RiskFlag.MAC_ADDR)
    if GPS_RE.search(check_payload):
        flags.append(RiskFlag.GPS_PRECISE)
    
    if re.search(r"latitude|longitude|location|coordinates|gps|geo_pos|lat|lon", check_payload, re.IGNORECASE):
        flags.append(RiskFlag.GPS_PRECISE)
    
    if re.search(r"serialno|serial_no|serial", check_payload, re.IGNORECASE):
        flags.append(RiskFlag.SERIAL_NO)
    
    patterns_credentials = [
        r"\b(pwd|password|passw|secret|apikey|access[_-]?key|auth[_-]?token|jwt|bearer)\b",
        
        # AWS Specific
        r"AKIA[0-9A-Z]{16}",
        r"aws[_-]?secret[_-]?access[_-]?key",
        
        # Azure Specific
        r"azure[_-]?storage[_-]?account[_-]?key",
        r"AccountKey=[a-zA-Z0-9+/=]+",

        # JWT
        r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}",
    ]

    combined_regex = "|".join(patterns_credentials)

    if re.search(combined_regex, check_payload, re.IGNORECASE):
        flags.append(RiskFlag.CREDENTIAL)
    if re.search(r"\b\d{8,}\b", check_payload):
        flags.append(RiskFlag.LONG_NUMERIC)
        
    # Return unique flags in order
    seen = set()
    uniq = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def compute_risk(flags: List[RiskFlag]) -> int:
    score = 0
    for f in flags:
        score += RISK_WEIGHTS.get(f, 1)
    return score

def heuristic_classify_multi(payload: str, struct: str) -> Dict[Category, float]:
    scores: Dict[Category, float] = {}
    s = payload.lower()
    
    # JSON Structural Checks
    if struct in ("json_object", "json_array", "json_like"):
        try:
            j = json.loads(payload)
        except Exception:
            j = None
        
        if isinstance(j, dict):
            keys = set(k.lower() for k in j.keys())
            
            # Fuzzy BLE/Events detection with weighted indicators
            ble_confidence = _detect_ble_event(j, keys)
            if ble_confidence > 0:
                scores[Category.EVENTS] = ble_confidence
            
            # AI inference
            if "objects" in keys and ("type" in keys or "confidence" in keys):
                scores[Category.AI_INFERENCE] = 0.99
            
            # Control indicators
            control_indicators = {"command", "cntrl", "control", "action", "lock", "unlock", 
                                 "brightness", "color_temp", "setpoint", "mode", "pump", "valve"}
            if any(k in keys for k in control_indicators):
                scores[Category.CONTROL] = max(scores.get(Category.CONTROL, 0), 0.8)
            
            # Events indicators
            event_indicators = {"event_type", "event", "alert", "alarm", "from_state", "to_state"}
            if any(k in keys for k in event_indicators):
                scores[Category.EVENTS] = max(scores.get(Category.EVENTS, 0), 0.75)
            
            # Telemetry
            if {"x", "y", "z"}.issubset(keys) or "position" in keys:
                scores[Category.TELEMETRY] = 0.95
        
        if isinstance(j, list):
            if any(isinstance(el, dict) and "registerNr" in el for el in j):
                scores[Category.TELEMETRY] = 0.95
    
    # Explicit Control Checks
    if s.strip() in {"on", "off", "true", "false"}:
        scores[Category.CONTROL] = max(scores.get(Category.CONTROL, 0), 0.9)
    
    if s.strip() in {"0", "1"}:
        scores[Category.CONTROL] = max(scores.get(Category.CONTROL, 0), 0.2)
    
    control_commands = {"start", "stop", "lock", "unlock", "open", "close", "on", "off"}
    words_in_payload = set(s.lower().split())
    words_with_separators = set()
    for word in words_in_payload:
        words_with_separators.add(word)
        for w in re.split(r'[_-]|(?=[A-Z])', word.lower()):
            if w:
                words_with_separators.add(w)
    
    if any(cmd in words_with_separators for cmd in control_commands):
        scores[Category.CONTROL] = max(scores.get(Category.CONTROL, 0), 0.7)
    
    # Simple Numeric Detection
    if struct == "numeric":
        try:
            val = float(payload.strip())
            if val not in (0.0, 1.0):
                scores[Category.TELEMETRY] = max(scores.get(Category.TELEMETRY, 0), 0.8)
        except Exception:
            pass
    
    # IP Address Detection
    if is_ip_address(payload):
        scores[Category.TELEMETRY] = max(scores.get(Category.TELEMETRY, 0), 0.85)
    
    # Base64/Encrypted Data Detection
    if is_base64_or_encrypted(payload):
        pass
    
    if struct in ("json_object", "json_array"):
        try:
            j = json.loads(payload)
            if isinstance(j, dict):
                values = list(j.values())
                if values and all(isinstance(v, bool) for v in values):
                    scores[Category.CONTROL] = max(scores.get(Category.CONTROL, 0), 0.75)
            elif isinstance(j, list):
                if j and all(isinstance(v, bool) for v in j):
                    scores[Category.CONTROL] = max(scores.get(Category.CONTROL, 0), 0.75)
        except Exception:
            pass
    
    is_csv_format = struct in ("csv_like", "keyvalue")
    is_json_format = struct in ("json_object", "json_array", "json_like")
    
    for cat in [c for c in Category if c != Category.UNSTRUCTURED]:
        keywords = KEYWORDS_MAP[cat]
        match_count = _count_keyword_matches(s, keywords)
        
        if match_count > 0:
            if cat == Category.EVENTS and (is_csv_format or is_json_format):
                matched_keywords = {kw for kw in keywords if kw in s}
                events_only_matches = matched_keywords & EVENTS_ONLY_KEYWORDS
                if not events_only_matches:
                    continue
            
            confidence = min(0.85, 0.5 + (match_count * 0.075))
            
            keyword_specificity = _get_keyword_specificity(cat)
            confidence *= keyword_specificity
            confidence = round(confidence, 3)
            
            scores[cat] = max(scores.get(cat, 0), confidence)
    
    if not scores:
        scores[Category.UNSTRUCTURED] = 0.0
    
    return scores