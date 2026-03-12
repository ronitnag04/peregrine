"""
categories.py

Single source of truth for gem5 OpClass ("Category" in our trace CSV) metadata:

- Coarse functional unit grouping (restricted to the FUs instantiated by
  Peregrine's `MyFUPool`).
- Base operation latency in cycles (defaults to 1 if not specified).

FU lists are derived from:
  peregrine-gem5/src/cpu/o3/FuncUnitConfig.py
and `MyFUPool` composition in:
  peregrine-gem5/configs/peregrine/peregrine.py
"""

from __future__ import annotations

from typing import Final, Literal

# Only the FU types used by peregrine-gem5/configs/peregrine/peregrine.py MyFUPool.
FUGroup = Literal[
    "int_alu",
    "int_mult_div",
    "fp_alu",
    "fp_mult_div",
    "read_port",
    "rdwr_port",
    "simd_unit",
]


# ----------------------------
# Base op latencies (cycles)
# ----------------------------

# Default is 1 if not present here.
_OPCLASS_LATENCY: Final[dict[str, int]] = {
    "IntMult": 3,
    "IntDiv": 20,
    "FloatAdd": 2,
    "FloatCmp": 2,
    "FloatCvt": 2,
    "Bf16Cvt": 2,
    "FloatMult": 4,
    "FloatMultAcc": 5,
    "FloatMisc": 3,
    "FloatDiv": 12,
    "FloatSqrt": 24,
}


def opclass_latency(op_class: str | None) -> int:
    c = (op_class or "").strip()
    if not c:
        return 1
    return int(_OPCLASS_LATENCY.get(c, 1))


# ----------------------------
# FU group mapping (OpClass)
# ----------------------------

_INT_ALU: Final[set[str]] = {"IntAlu"}
_INT_MD: Final[set[str]] = {"IntMult", "IntDiv"}

_FP_ALU: Final[set[str]] = {"FloatAdd", "FloatCmp", "FloatCvt", "Bf16Cvt"}
_FP_MD: Final[set[str]] = {"FloatMult", "FloatMultAcc", "FloatMisc", "FloatDiv", "FloatSqrt"}

_SIMD_UNIT: Final[set[str]] = {
    "SimdAdd",
    "SimdAddAcc",
    "SimdAlu",
    "SimdCmp",
    "SimdCvt",
    "SimdMisc",
    "SimdMult",
    "SimdMultAcc",
    "SimdMatMultAcc",
    "SimdShift",
    "SimdShiftAcc",
    "SimdDiv",
    "SimdSqrt",
    "SimdFloatAdd",
    "SimdFloatAlu",
    "SimdFloatCmp",
    "SimdFloatCvt",
    "SimdFloatDiv",
    "SimdFloatMisc",
    "SimdFloatMult",
    "SimdFloatMultAcc",
    "SimdFloatMatMultAcc",
    "SimdFloatSqrt",
    "SimdReduceAdd",
    "SimdReduceAlu",
    "SimdReduceCmp",
    "SimdFloatReduceAdd",
    "SimdFloatReduceCmp",
    "SimdExt",
    "SimdFloatExt",
    "SimdConfig",
    "SimdDotProd",
    "SimdAes",
    "SimdAesMix",
    "SimdSha1Hash",
    "SimdSha1Hash2",
    "SimdSha256Hash",
    "SimdSha256Hash2",
    "SimdShaSigma2",
    "SimdShaSigma3",
    "SimdSha3",
    "SimdSm4e",
    "SimdCrc",
    "SimdBf16Add",
    "SimdBf16Cmp",
    "SimdBf16Cvt",
    "SimdBf16DotProd",
    "SimdBf16MatMultAcc",
    "SimdBf16Mult",
    "SimdBf16MultAcc",
}

# Read-only memory ops (ReadPort)
_READ_PORT: Final[set[str]] = {
    "MemRead",
    "FloatMemRead",
    "SimdUnitStrideLoad",
    "SimdUnitStrideMaskLoad",
    "SimdUnitStrideSegmentedLoad",
    "SimdStridedLoad",
    "SimdIndexedLoad",
    "SimdUnitStrideFaultOnlyFirstLoad",
    "SimdUnitStrideSegmentedFaultOnlyFirstLoad",
    "SimdWholeRegisterLoad",
    "SimdStrideSegmentedLoad",
}

# Read/write memory ops (RdWrPort). Note this includes MemRead too (gem5 model),
# but we classify "MemRead" as ReadPort first so it remains read_port.
_RDWR_PORT: Final[set[str]] = {
    "MemRead",
    "MemWrite",
    "FloatMemRead",
    "FloatMemWrite",
    "SimdUnitStrideLoad",
    "SimdUnitStrideStore",
    "SimdUnitStrideMaskLoad",
    "SimdUnitStrideMaskStore",
    "SimdUnitStrideSegmentedLoad",
    "SimdUnitStrideSegmentedStore",
    "SimdStridedLoad",
    "SimdStridedStore",
    "SimdIndexedLoad",
    "SimdIndexedStore",
    "SimdUnitStrideFaultOnlyFirstLoad",
    "SimdUnitStrideSegmentedFaultOnlyFirstLoad",
    "SimdWholeRegisterLoad",
    "SimdWholeRegisterStore",
    "SimdStrideSegmentedLoad",
    "SimdStrideSegmentedStore",
}


def opclass_to_fu_group(op_class: str | None) -> FUGroup:
    """
    Map gem5 OpClass string (trace CSV "Category") to one of the FU groups we
    actually instantiate in Peregrine's MyFUPool.
    """
    c = (op_class or "").strip()
    if not c:
        raise ValueError("Empty OpClass/Category in trace row")

    # Check read_port before rdwr_port (since rdwr_port includes MemRead too).
    if c in _READ_PORT:
        return "read_port"
    if c in _RDWR_PORT:
        return "rdwr_port"
    if c in _INT_MD:
        return "int_mult_div"
    if c in _INT_ALU:
        return "int_alu"
    if c in _FP_MD:
        return "fp_mult_div"
    if c in _FP_ALU:
        return "fp_alu"
    if c in _SIMD_UNIT:
        return "simd_unit"

    return "other"

