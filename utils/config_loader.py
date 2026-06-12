from pathlib import Path
from typing import Any, Dict, List

import yaml

from config.settings import settings


def load_match_config(yaml_path: Path) -> Dict[str, Any]:
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_match_configs() -> List[Path]:
    root = settings.matches_dir
    return sorted(p for p in root.rglob("*.yaml") if not p.name.startswith("_"))
