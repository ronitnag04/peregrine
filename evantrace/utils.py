import numpy as np
from numpy.typing import NDArray

"""
np.where wrapper to prevent annoying indexing. Returns the
index of the first occurrence of key in arr. Can be used
with any numpy type
"""
def where(arr: NDArray[np.generic], key: np.generic) -> int | None:
    indices = np.where(arr == key)[0]
    if len(indices) == 0:
        return None
    return indices[0]