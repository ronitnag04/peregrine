"""
models.py — Python Resource enum and re-exports.

All lists, dicts, and constants that were previously defined here
(RESOURCE_FILES, DOUBLE_PARAM_RESOURCES, RESOURCE_PARAM_SPECS, etc.)
now live in registry.py and are derived from registry.yaml.

Config is auto-generated from registry.yaml by gen_registry.py → config_gen.py.
"""

from enum import Enum

import registry
from config_gen import Config  # noqa: F401 — re-exported for callers


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


