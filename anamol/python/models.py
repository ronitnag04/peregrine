"""
models.py — Python Config dataclass and Resource enum.

All lists, dicts, and constants that were previously defined here
(RESOURCE_FILES, DOUBLE_PARAM_RESOURCES, RESOURCE_PARAM_SPECS, etc.)
now live in registry.py and are derived from registry.yaml.

This file keeps only the Config dataclass and the Resource enum,
both of which are sourced from registry.py.
"""

from dataclasses import dataclass
from enum import Enum

import registry


# Resource enum — values are the canonical names used as lookup keys and
# .npy filename stems. Derived from registry.yaml resource order.
Resource = Enum(
    "Resource",
    {r.name.upper(): r.name for r in registry.RESOURCES},
)


def resource_key(r):
    """Return the string key for a Resource enum member or string."""
    if isinstance(r, Resource):
        return r.value
    return r


# Re-export the derived constants from registry so callers that imported
# them from models.py continue to work.
RESOURCE_FILES = registry.RESOURCE_FILES
DOUBLE_PARAM_RESOURCES = registry.DOUBLE_PARAM_RESOURCES
RESOURCE_PARAM_SPECS = registry.RESOURCE_PARAM_SPECS
RESOURCE_PARAM_SPECS_BY_KEY = registry.RESOURCE_PARAM_SPECS  # same dict


def get_resource_param_spec(r):
    """Return param name list for resource r (accepts Resource or string)."""
    return registry.RESOURCE_PARAM_SPECS.get(resource_key(r), ["param"])


@dataclass
class Config:
    # Defaults come from registry.yaml — edit there, not here.
    rob_size: int = registry.PARAMS_BY_NAME["rob_size"].default
    commit_width: int = registry.PARAMS_BY_NAME["commit_width"].default
    load_queue_size: int = registry.PARAMS_BY_NAME["load_queue_size"].default
    store_queue_size: int = registry.PARAMS_BY_NAME["store_queue_size"].default
    alu_issue_width: int = registry.PARAMS_BY_NAME["alu_issue_width"].default
    alu_mul_issue_width: int = registry.PARAMS_BY_NAME["alu_mul_issue_width"].default
    alu_div_issue_width: int = registry.PARAMS_BY_NAME["alu_div_issue_width"].default
    fp_issue_width: int = registry.PARAMS_BY_NAME["fp_issue_width"].default
    ls_issue_width: int = registry.PARAMS_BY_NAME["ls_issue_width"].default
    num_ls_pipes: int = registry.PARAMS_BY_NAME["num_ls_pipes"].default
    num_load_pipes: int = registry.PARAMS_BY_NAME["num_load_pipes"].default
    fetch_width: int = registry.PARAMS_BY_NAME["fetch_width"].default
    decode_width: int = registry.PARAMS_BY_NAME["decode_width"].default
    rename_width: int = registry.PARAMS_BY_NAME["rename_width"].default
    num_fetch_buffers: int = registry.PARAMS_BY_NAME["num_fetch_buffers"].default
    max_icache_fills: int = registry.PARAMS_BY_NAME["max_icache_fills"].default
    branch_predictor: int = registry.PARAMS_BY_NAME["branch_predictor"].default
    misprediction_percent: int = registry.PARAMS_BY_NAME["misprediction_percent"].default
    l1d_cache_kb: int = registry.PARAMS_BY_NAME["l1d_cache_kb"].default
    l1i_cache_kb: int = registry.PARAMS_BY_NAME["l1i_cache_kb"].default
    l2_cache_kb: int = registry.PARAMS_BY_NAME["l2_cache_kb"].default
    l1d_stride_prefetch: int = registry.PARAMS_BY_NAME["l1d_stride_prefetch"].default
