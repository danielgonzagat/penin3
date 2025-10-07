#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apis.litellm_wrapper import LiteLLMWrapper
from config.settings import API_KEYS, API_MODELS, DATA_DIR

def main():
    prompt = os.environ.get('SMOKE_PROMPT', 'Responda OK.')
    max_tokens = int(os.environ.get('SMOKE_MAX_TOKENS', '12'))
    wrapper = LiteLLMWrapper(API_KEYS, API_MODELS)
    out = wrapper.call_all_models_robust(prompt, max_tokens=max_tokens)
    # Persist results
    out_path = Path(DATA_DIR) / 'api_smoke.json'
    try:
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception:
        pass
    # Print concise summary
    print(json.dumps({k: (v[:80] if isinstance(v, str) else v) for k, v in out.items()}, ensure_ascii=False))

if __name__ == '__main__':
    main()
