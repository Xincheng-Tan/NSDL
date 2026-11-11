import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "verl", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


import json
import pytest


@pytest.fixture(autouse=True)
def _dummy_api_keys(tmp_path_factory, monkeypatch):
    """`nsdl.api_backend` reads api_keys.json (git-ignored); tests never
    contact a real endpoint, so point it at a throwaway file."""
    try:
        import nsdl.api_backend as ab
    except Exception:  # pragma: no cover - module may be unimportable in slim envs
        return
    line = {"base_url": "https://relay.example.com", "openai_base_url": "https://relay.example.com/v1",
            "OPENAI_KEY": "sk-dummy", "CLAUDE_KEY": "sk-dummy"}
    f = tmp_path_factory.mktemp("keys") / "api_keys.json"
    f.write_text(json.dumps({"RELAY_A": line, "RELAY_A2": dict(line)}))
    monkeypatch.setattr(ab, "_KEYS", f)
