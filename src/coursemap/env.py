import os
from pathlib import Path


def read_dotenv(path: Path = Path(".env")) -> dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def read_key_file(path: Path) -> str | None:
    if not path.exists():
        return None
    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            value = path.read_text(encoding=encoding).strip()
            return value or None
        except UnicodeDecodeError:
            continue
    return None


def resolve_secret(name: str, file_candidates: list[Path] | None = None) -> str | None:
    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value
    dotenv_value = read_dotenv().get(name, "").strip()
    if dotenv_value:
        return dotenv_value
    for path in file_candidates or []:
        value = read_key_file(path)
        if value:
            return value
    return None
