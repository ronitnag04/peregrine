"""
registry.py — Python-side reader for registry.yaml.

This is the single source of truth for all resource and parameter definitions
on the Python side. All other Python modules import from here instead of
defining their own lists/dicts.

The YAML is loaded once at import time from the default path
(anamol/registry.yaml, resolved relative to this file's location).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "PyYAML is required. Install with: uv pip install pyyaml"
    ) from e


# Default path: python/registry.py → ../registry.yaml
_DEFAULT_YAML = Path(__file__).parent.parent / "registry.yaml"


@dataclass(frozen=True)
class ParamDef:
    name: str       # snake_case — matches Config field name
    min_val: int
    max_val: int
    step: str       # "base2" or "linear"
    default: int
    enabled: bool


@dataclass(frozen=True)
class ResourceDef:
    name: str           # canonical name — lookup key AND thr_{name}.npy stem
    params: List[str]   # ordered param names that drive this resource's sweep
    enabled: bool


def _load(yaml_path: Path) -> tuple[List[ParamDef], List[ResourceDef]]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    params = [
        ParamDef(
            name=p["name"],
            min_val=p["min"],
            max_val=p["max"],
            step=p["step"],
            default=p["default"],
            enabled=p.get("enabled", True),
        )
        for p in data.get("params", [])
    ]

    resources = [
        ResourceDef(
            name=r["name"],
            params=list(r["params"]),
            enabled=r.get("enabled", True),
        )
        for r in data.get("resources", [])
    ]

    return params, resources


# Load once at import time
PARAMS: List[ParamDef]
RESOURCES: List[ResourceDef]
PARAMS, RESOURCES = _load(_DEFAULT_YAML)

# Fast lookup dicts
PARAMS_BY_NAME: Dict[str, ParamDef] = {p.name: p for p in PARAMS}
RESOURCES_BY_NAME: Dict[str, ResourceDef] = {r.name: r for r in RESOURCES}

# Filtered views
ENABLED_PARAMS: List[ParamDef] = [p for p in PARAMS if p.enabled]
ENABLED_RESOURCES: List[ResourceDef] = [r for r in RESOURCES if r.enabled]

# Derived helpers that the rest of the codebase previously computed manually
RESOURCE_FILES: List[str] = [f"thr_{r.name}.npy" for r in ENABLED_RESOURCES]

DOUBLE_PARAM_RESOURCES: frozenset = frozenset(
    r.name for r in RESOURCES if len(r.params) > 1
)

# {resource_name: [param_name, ...]} — same semantics as old RESOURCE_PARAM_SPECS_BY_KEY
RESOURCE_PARAM_SPECS: Dict[str, List[str]] = {r.name: r.params for r in RESOURCES}
