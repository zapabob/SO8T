from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    creds_path = Path(r"C:\Users\downl\.gemini\oauth_creds.json")
    config_dir = Path(r"C:\Users\downl\.config")
    config_path = config_dir / "gemini-cli.toml"

    data = json.loads(creds_path.read_text(encoding="utf-8"))
    token = data.get("access_token")
    if not token:
        raise RuntimeError("access_token not found in oauth_creds.json")

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(f'token = "{token}"\n', encoding="utf-8")


if __name__ == "__main__":
    main()
