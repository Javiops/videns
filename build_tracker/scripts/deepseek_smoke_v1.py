r"""Connectivity check for the DeepSeek API — lists the model IDs this key can
call, then optionally runs one minimal completion.

Nothing in the pipeline depends on this. It exists to answer one question
before any integration is written: which model IDs actually exist on the
account, since the ID in the docs and the ID the API serves are not always
the same string.

    .venv\Scripts\python.exe scripts\deepseek_smoke_v1.py
    .venv\Scripts\python.exe scripts\deepseek_smoke_v1.py --model deepseek-chat
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://api.deepseek.com"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="if given, send one short test prompt to this model id")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="completion budget; reasoning models spend this on hidden reasoning "
                         "first, so a small cap returns empty content (default: 512)")
    args = ap.parse_args()

    load_dotenv()
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        print("DEEPSEEK_API_KEY not set. Add it to .env (see .env.example).", file=sys.stderr)
        return 2

    client = OpenAI(api_key=key, base_url=BASE_URL)

    print(f"models visible to this key at {BASE_URL}:")
    for m in client.models.list().data:
        print(f"  {m.id}")

    if args.model:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=args.max_tokens,
            stream=False,
        )
        print(f"\n{args.model} -> {resp.choices[0].message.content!r}")
        print(f"usage: {resp.usage}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
