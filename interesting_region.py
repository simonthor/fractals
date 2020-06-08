import numpy as np
from numba import njit


@njit
def entropy(array: np.ndarray, base: float = None) -> np.float64:
    """Calculate the entropy of a distribution for given probability values. Code inspired by [1]_ and [2]_.
    Parameters
    -------
    array : np.ndarray
        Defines all  (discrete) distribution. ``pk[i]`` is the (possibly
        unnormalized) probability of event ``i``.
    base : float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).

    Returns
    -------
    S : float
        The calculated entropy.

    References
    -------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.entr.html
    """
    probability_counts = np.bincount(array.ravel())
    p_norm = probability_counts / probability_counts.sum()
    entropy = np.zeros(p_norm.shape) - np.Inf
    above_zero = p_norm > 0
    entropy[above_zero] = -p_norm[above_zero] * np.log(p_norm[above_zero])
    is_zero = p_norm == 0
    entropy[is_zero] = 0
    S = entropy.sum()

    if base is not None:
        S /= np.log(base)

    return S


@njit
def find_interesting_region(plane: np.ndarray, zoom_factor: float, iterations: np.int64) -> np.ndarray:
    a = np.empty(2)
    np.round(np.array(plane.shape) / zoom_factor, 0, a)
    x_factor, y_factor = a.astype(np.int64)
    variance = 0
    segment_index_range = np.ones((2, 2)) * -1

    for x in range(plane.shape[0] - x_factor):
        for y in range(plane.shape[1] - y_factor):
            segment = plane[x:x + x_factor + 1, y:y + y_factor + 1]
            segment_variance = entropy(segment.astype(np.int64))
            if segment_variance > variance:
                variance = segment_variance
                segment_index_range[:] = np.array([[x, x + x_factor], [y, y + y_factor]])

    return segment_index_range.astype(np.int64)


@njit
def faster_unique(array: np.ndarray, max_val: np.int64) -> np.int64:
    unique_vals = np.zeros(max_val, dtype=np.uint8)
    unique_vals[array.ravel()] = 1
    return unique_vals.sum()


@njit
def new_variance(array: np.ndarray, iterations: np.int64) -> np.float64:
    vertical = np.diff(array)
    horizontal = np.diff(array.T)
    return faster_unique(vertical, iterations) + faster_unique(horizontal, iterations)


@njit
def max_var_segment(plane: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Identifies the segment of a 2D array which is the most interesting.

    Parameters
    ----------
    plane : numpy.ndarray
        Values that will be examined.
    zoom_factor : int
        How small each segment should be. If `zoom_factor = 2` and `plane` has shape 100x100, then each segment is 50x50.

    Returns
    ----------
    segment_index : numpy.ndarray
        The start and end index of the segment with largest variance, e.g. [0 10].
        The first returned parameter is the x (row) range and the second parameter is the y (column) range.

    Notes
    ----------
    The algorithm currently used for calcualting how interesting a region is is shannon entropy [1]_.

    RefreSee https://en.wiktionary.org/wiki/Shannon_entropy
    """
    a = np.empty(2)
    np.round(np.array(plane.shape) / zoom_factor, 0, a)
    x_factor, y_factor = a.astype(np.int64)
    variance = 0
    segment_index_range = np.ones((2, 2)) * -1
    for x in range(plane.shape[0] - x_factor):
        for y in range(plane.shape[1] - y_factor):
            segment = plane[x:x + x_factor + 1, y:y + y_factor + 1]
            segment_variance = ((segment - segment.mean())**2).sum()
            if segment_variance > variance:
                variance = segment_variance
                segment_index_range[:] = np.array([[x, x + x_factor], [y, y + y_factor]])

    return segment_index_range.astype(np.int64)


