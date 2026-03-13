from __future__ import annotations

import os
import sys
from pathlib import Path

import ollama


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Verify that Ollama is reachable and the configured model responds."""

    model = os.getenv("OLLAMA_MODEL", "llama3")
    print(f"Testing Ollama connection with model: {model}")

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "Reply with one short sentence confirming Ollama is working."}],
        )
    except Exception as error:
        raise SystemExit(
            "Ollama is not reachable. Make sure Ollama is installed, running, and the selected model is available locally."
        ) from error

    print("Ollama response:")
    print(response["message"]["content"].strip())


if __name__ == "__main__":
    main()
