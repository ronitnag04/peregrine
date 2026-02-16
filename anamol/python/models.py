from dataclasses import dataclass
from enum import Enum

RESOURCE_FILES = [
    "thr_rob.npy",
    "thr_load_queue.npy",
    "thr_store_queue.npy",
    "thr_alu_issue.npy",
    "thr_fp_issue.npy",
    "thr_ls_issue.npy",
    "thr_load_ls_pipes_lower.npy",
    "thr_load_ls_pipes_upper.npy",
    "thr_icache_fills.npy",
    "thr_fetch_buffers.npy",
]


# new: enum of resource keys (values match the keys used in data)
class Resource(Enum):
    ROB = "rob"
    LOAD_QUEUE = "load_queue"
    STORE_QUEUE = "store_queue"
    ALU_ISSUE = "alu_issue"
    FP_ISSUE = "fp_issue"
    LS_ISSUE = "ls_issue"
    LOAD_LS_PIPES_LOWER = "load_ls_pipes_lower"
    LOAD_LS_PIPES_UPPER = "load_ls_pipes_upper"
    ICACHE_FILLS = "icache_fills"
    FETCH_BUFFERS = "fetch_buffers"


def resource_key(r):
    """Return the string key for a resource enum or string (backwards compatible)."""
    if isinstance(r, Resource):
        return r.value
    return r


DOUBLE_PARAM_RESOURCES = {
    "load_ls_pipes_lower",
    "load_ls_pipes_upper",
}

# Parameter name specifications per resource (use Resource enum keys).
RESOURCE_PARAM_SPECS = {
    Resource.ROB: ["ROB_SIZE"],
    Resource.LOAD_QUEUE: ["LOAD_QUEUE_SIZE"],
    Resource.STORE_QUEUE: ["STORE_QUEUE_SIZE"],
    Resource.ALU_ISSUE: ["ALU_ISSUE_WIDTH"],
    Resource.FP_ISSUE: ["FP_ISSUE_WIDTH"],
    Resource.LS_ISSUE: ["LS_ISSUE_WIDTH"],
    Resource.LOAD_LS_PIPES_LOWER: ["NUM_LS_PIPES", "NUM_LOAD_PIPES"],
    Resource.LOAD_LS_PIPES_UPPER: ["NUM_LS_PIPES", "NUM_LOAD_PIPES"],
    Resource.ICACHE_FILLS: ["MAX_ICACHE_FILLS"],
    Resource.FETCH_BUFFERS: ["NUM_FETCH_BUFFERS"],
}

# Backwards-compatible mapping keyed by string keys and a small accessor.
RESOURCE_PARAM_SPECS_BY_KEY = {
    resource_key(r): spec for r, spec in RESOURCE_PARAM_SPECS.items()
}


def get_resource_param_spec(r):
    """Return param name list for resource r (accepts Resource or string)."""
    return RESOURCE_PARAM_SPECS_BY_KEY.get(resource_key(r), ["param"])


@dataclass
class Config:
    # Core/commit
    rob_size: int = 128  # [1..1024]
    commit_width: int = 4  # [1..12]

    # Queues
    load_queue_size: int = 64  # [1..256]
    store_queue_size: int = 64  # [1..256]

    # Issue widths
    alu_issue_width: int = 4  # [1..8]
    fp_issue_width: int = 4  # [1..8]
    ls_issue_width: int = 2  # [1..8]

    # Pipelines
    num_ls_pipes: int = 2  # [1..8]
    num_load_pipes: int = 2  # [0..8]

    # Frontend widths
    fetch_width: int = 4  # [1..12]
    decode_width: int = 4  # [1..12]
    rename_width: int = 4  # [1..12]

    # Frontend buffers and fills
    num_fetch_buffers: int = 4  # [1..8]
    max_icache_fills: int = 8  # [1..32]

    # Branch predictor and accuracy
    branch_predictor: int = 0  # 0/1 (linear)
    misprediction_percent: int = 5  # [0..100]

    # Caches (KB)
    l1d_cache_kb: int = 64  # [16..256]
    l1i_cache_kb: int = 64  # [16..256]
    l2_cache_kb: int = 1024  # [512..4096]

    # Prefetch
    l1d_stride_prefetch: int = 1  # 0/1 (linear)
