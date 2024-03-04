import numpy as np
from typing import Tuple


def g_from_nv(n_array: np.array, v_array: np.array) -> np.ndarray:
    assert len(n_array.shape) == len(v_array.shape) == 1

    w = np.array([[-7.497527849096219, -7.49752916467739],
                  [0.07842101471405442, -0.07842100095362642]])
    mean = np.array([[1.6426209211349487, 48.8505973815918]])

    return np.matmul((np.stack((n_array, v_array), axis=-1) - mean), w)
