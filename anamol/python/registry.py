"""
registry.py — Python-side reader for registry.yaml.

This is the single source of truth for all resource and parameter definitions
on the Python side. All other Python modules import from here instead of
defining their own lists/dicts.

The YAML is loaded once at import time from the default path
(anamol/registry.yaml, resolved relative to this file's location).
"""

from dataclasses import dataclass, field
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
    min_val: int | float
    max_val: int | float
    step: str | list[int]  # "base2", "linear", or explicit list e.g. [0, 4]
    default: int | float
    enabled: bool
    # gem5 field name(s):
    #   str       → direct 1:1 mapping (e.g. "rob_size")
    #   list[str] → sum of those gem5 fields (e.g. ls_issue_width)
    #   None      → no gem5 counterpart (name_gem5 absent or "none")
    name_gem5: str | list[str] | None = None
    # "int" (default) or "float" — controls Config field type and sampling
    param_type: str = "int"


@dataclass(frozen=True)
class ResourceDef:
    name: str           # canonical name — lookup key AND thr_{name}.npy stem
    params: List[str]   # ordered param names that drive this resource's sweep
    enabled: bool
    # Latency types this resource depends on — subset of {"exe", "fetch"}.
    # "exe"   → calls resp_cycle() → instr.exe_latency
    # "fetch" → uses instr.fetch_latency directly
    # []      → instruction counts only; computed once across all cache configs
    latency_dependent: List[str] = field(default_factory=list)


def _load(yaml_path: Path) -> tuple[List[ParamDef], List[ResourceDef]]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    params = []
    for p in data.get("params", []):
        step = p["step"]
        raw_gem5 = p.get("name_gem5")
        if raw_gem5 == "none":
            name_gem5 = None
        else:
            name_gem5 = raw_gem5  # str, list[str], or None
        param_type = p.get("type", "int")
        if isinstance(step, list):
            vals = list(step)
            params.append(ParamDef(
                name=p["name"],
                min_val=min(vals),
                max_val=max(vals),
                step=vals,
                default=p["default"],
                enabled=p.get("enabled", True),
                name_gem5=name_gem5,
                param_type=param_type,
            ))
        else:
            params.append(ParamDef(
                name=p["name"],
                min_val=p["min"],
                max_val=p["max"],
                step=step,
                default=p["default"],
                enabled=p.get("enabled", True),
                name_gem5=name_gem5,
                param_type=param_type,
            ))

    resources = [
        ResourceDef(
            name=r["name"],
            params=list(r["params"]),
            enabled=r.get("enabled", True),
            latency_dependent=list(r.get("latency_dependent", [])),
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

LATENCY_DEPENDENT_RESOURCES: frozenset = frozenset(
    r.name for r in RESOURCES if r.latency_dependent
)

EXE_LATENCY_RESOURCES: frozenset = frozenset(
    r.name for r in RESOURCES if "exe" in r.latency_dependent
)

FETCH_LATENCY_RESOURCES: frozenset = frozenset(
    r.name for r in RESOURCES if "fetch" in r.latency_dependent
)

# {resource_name: [param_name, ...]} — same semantics as old RESOURCE_PARAM_SPECS_BY_KEY
RESOURCE_PARAM_SPECS: Dict[str, List[str]] = {r.name: r.params for r in RESOURCES}
