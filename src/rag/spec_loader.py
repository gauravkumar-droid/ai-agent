from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_api_specs(spec_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all YAML specs in the directory into dictionaries.
    """

    spec_dir = Path(spec_dir)
    if not spec_dir.exists():
        raise FileNotFoundError(f"Spec directory {spec_dir} does not exist")

    specs: List[Dict[str, Any]] = []
    for path in sorted(spec_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh) or {}
            specs.append(spec)

    if not specs:
        raise ValueError(f"No YAML specs found in {spec_dir}")

    return specs
