"""Standard-library API regression tests; run with API_BASE_URL=... python3 test_api.py."""
import copy
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:3001").rstrip("/")


def request(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status, response.read().decode(), time.perf_counter()-start
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(), time.perf_counter()-start


def wait_ready(seconds=180):
    end = time.monotonic()+seconds
    while time.monotonic() < end:
        try:
            if request("GET", "/readyz")[0] == 200:
                return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    return False


def correct_result(status, body, expected):
    if status != 200:
        return False
    try:
        result = json.loads(body)
        return type(result.get("prediction")) is int and result == expected
    except (ValueError, AttributeError):
        return False


def rejects_field(status, body, field):
    if status not in (400, 422):
        return False
    try:
        details = json.loads(body).get("detail", [])
        return isinstance(details, list) and any(
            isinstance(item, dict) and field in item.get("loc", []) for item in details)
    except (ValueError, TypeError, AttributeError):
        return False


def main():
    fixtures = json.loads((HERE / "api_fixtures.json").read_text())
    results = []
    def check(name, passed, detail=""):
        results.append(bool(passed))
        print(f"{'PASS' if passed else 'FAIL'}: {name}" + (f" | {detail}" if detail else ""))
    print("Testing:", BASE_URL)
    check("API readiness", wait_ready())
    if not results[0]:
        print("API did not become ready")
        return 1
    for case in fixtures:
        status, body, elapsed = request("POST", "/predict", case["request"])
        check(case["name"] + " golden prediction", correct_result(status, body, case["expected"]),
              f"HTTP {status}, {elapsed*1000:.1f} ms" + (f" body={body[:300]}" if status != 200 else ""))
    sample = fixtures[0]["request"]
    changes = [("negative age", "age", -5), ("age above limit", "age", 121),
               ("unknown symptom", "polyuria", "Maybe"), ("unknown gender", "gender", "Other"),
               ("extra field", "not_a_feature", "unexpected")]
    for name, field, value in changes:
        payload = copy.deepcopy(sample); payload["input_data"][field] = value
        status, body, _ = request("POST", "/predict", payload)
        check(name, rejects_field(status, body, field), f"HTTP {status}: {body[:250]}")
    payload = copy.deepcopy(sample); del payload["input_data"]["age"]
    status, body, _ = request("POST", "/predict", payload)
    check("missing age", rejects_field(status, body, "age"), f"HTTP {status}")
    status, _, _ = request("GET", "/predict")
    check("GET prediction rejected", status == 405, f"HTTP {status}")
    repeated = [request("POST", "/predict", sample) for _ in range(5)]
    check("five repeat predictions consistent", all(correct_result(s,b,fixtures[0]["expected"]) for s,b,_ in repeated))
    
    # Test application version endpoint
    status, body, elapsed = request("POST", "/version", {})

    expected_version = {
        "application": "Diabetes MLOps",
        "version": "1.1.0",
        "framework": "BentoML",
        "purpose": "Educational"
    }

    try:
        version_valid = (
            status == 200
            and json.loads(body) == expected_version
        )
    except (ValueError, TypeError):
        version_valid = False

    check(
        "Application version endpoint",
        version_valid,
        f"HTTP {status}, {elapsed*1000:.1f} ms | {body[:300]}"
    )

    print(f"RESULT: {sum(results)}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
