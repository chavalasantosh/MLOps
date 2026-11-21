import hashlib, json
from pathlib import Path
p = Path(__file__).resolve().parent
m = json.loads((p / 'model_manifest.json').read_text())
assert m['educational_gate_passed'] is True
assert hashlib.sha256((p / 'diabetes_model.bentomodel').read_bytes()).hexdigest() == m['model_archive_sha256'], 'Model artifact checksum mismatch'
assert len(json.loads((p / 'api_fixtures.json').read_text())) == 3
print('Model manifest and fixture integrity verified:', m['bento_tag'])
