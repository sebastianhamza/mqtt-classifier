from enum import Enum


class Category(str, Enum):
    CONTROL = "control"
    DEVICE_HEALTH = "device_health"
    EVENTS = "events"
    TELEMETRY = "telemetry"
    AI_INFERENCE = "ai_inference"
    UNSTRUCTURED = "unstructured"


class RiskFlag(str, Enum):
    CREDENTIAL = "credential"
    BIOMETRIC = "biometric"
    
    EMAIL = "email"
    GPS_PRECISE = "gps_precise"
    VISUAL_DATA = "visual_data"
    ACTUATION = "actuation"
    AUTH_TOKEN = "auth_token"

    IPV4_PUBLIC = "ipv4_public"
    IPV6 = "ipv6"
    MAC_ADDR = "mac_addr"
    BEACON_ID = "beacon_id"
    USER_ID = "user_id"

    WIFI_SSID = "wifi_ssid"
    SERIAL_NO = "serial_no"
    HOSTNAME = "hostname"
    FIRMWARE = "firmware"
    LONG_NUMERIC = "long_numeric"
    IPV4_PRIVATE = "ipv4_private"

    IPV4_LOCAL = "ipv4_local"
    OBIS_CODE = "obis_code"
    TIMESTAMP = "timestamp"

RISK_WEIGHTS = {
    RiskFlag.CREDENTIAL: 5,
    RiskFlag.BIOMETRIC: 5,

    RiskFlag.EMAIL: 4,
    RiskFlag.GPS_PRECISE: 4,
    RiskFlag.VISUAL_DATA: 4,
    RiskFlag.ACTUATION: 4,
    RiskFlag.AUTH_TOKEN: 4,

    RiskFlag.IPV4_PUBLIC: 3,
    RiskFlag.IPV6: 3,
    RiskFlag.MAC_ADDR: 3,
    RiskFlag.BEACON_ID: 3,
    RiskFlag.USER_ID: 3,

    RiskFlag.WIFI_SSID: 2,
    RiskFlag.SERIAL_NO: 2,
    RiskFlag.HOSTNAME: 2,
    RiskFlag.FIRMWARE: 2,
    RiskFlag.LONG_NUMERIC: 2,

    RiskFlag.IPV4_LOCAL: 1,
    RiskFlag.OBIS_CODE: 1,
    RiskFlag.TIMESTAMP: 1,
}

CATEGORY_KEYWORDS = {
    Category.TELEMETRY: [
        "voltage", "power", "amp", "battery", "temp", "temperature", "frequency", "wh", "watt", "soc",
        "charge", "discharge", "position", "coordinate", "robot", "axis",
        "ip", "address", "gateway", "mac", "network", "udp", "tcp",
        "humidity", "pressure", "co2", "light", "distance", "speed",
    ],
    Category.EVENTS: [
        "event", "motion", "trigger", "notification", "alert", "alarm",
        "door_open", "door_close", "motion_detected", "event_type",
        "from_state", "to_state", "state_transition", "changed", "change",
        "ble", "beacon", "ibeacon", "advertisement", "adv",
        "update", "startup", "shutdown", "reboot",
    ],
    Category.AI_INFERENCE: [
        "person", "object", "camera", "detect", "detection", "confidence", "frame",
        "classification", "prediction", "inference", "label", "bounding", "box",
        "anomaly", "pose", "keypoint", "class",
    ],
    Category.DEVICE_HEALTH: [
        "warn", "warning", "error", "fail", "failure", "overflow",
        "exception", "traceback", "log", "critical", "fault", "sensor fault",
        "memory", "disk", "cpu", "status", "health", "degraded",
    ],
    Category.CONTROL: [
        "command", "setpoint", "actuator", "move", "start", "stop",
        "enable", "disable", "configure", "control", "adjust", "override",
        "action", "brightness", "color", "temperature_setpoint", "mode",
        "on", "off", "lock", "unlock", "open", "close", "pump", "valve",
    ],
}