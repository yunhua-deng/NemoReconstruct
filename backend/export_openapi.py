from __future__ import annotations

import json
from pathlib import Path

from app.main import app


def main() -> None:
    output_path = Path(__file__).resolve().parent.parent / "docs" / "openapi.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
    print(f"Wrote OpenAPI schema to {output_path}")


if __name__ == "__main__":
    main()
