import numpy as np
from matplotlib import pyplot as plt
from typing import Callable, Tuple
from numba import njit


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
     iteration   - Required  : current iteration (Int)
     total       - Required  : total iterations (Int)
     prefix      - Optional  : prefix string (Str)
     suffix      - Optional  : suffix string (Str)
     decimals    - Optional  : positive number of decimals in percent complete (Int)
     length      - Optional  : character length of bar (Int)
     fill        - Optional  : bar fill character (Str)
     printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    Source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def generate_escape_time(plane: np.ndarray, iterations: int,
                         et_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                         *et_f_args, **et_f_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Generates an escape time fractal.

    Parameters
    ----------
    plane : numpy.ndarray
        The coordinates in the complex plane that will be used to generate the mandelbrot set.
    iterations : int
        Maximum number of iterations that the mandelbrot formula will be used on a given number.
    et_function : function
        The function used to calculate the z value of each point in the complex plane.
        Must at least accept (but do not have to use) 2 input arguments of the complex plane and the current z values.
    *et_f_args : any, optional
        Other input arguments for et_function.
    **et_f_kwargs : any, optional
        Other input keyword arguments for et_function.

    Returns
    ----------
    iteration_counter : numpy.ndarray
        Number of iterations before |z| >= 2. Same shape as plane.
    z_values : numpy.ndarray
        The values of the complex numbers right after |z| >= 2. Same shape as plane.

    Notes
    ----------
    The mandelbrot set uses the formula :math:`z_i = z_{i-1}^2+c`
    """

    z_values = np.copy(plane)
    iteration_counter = np.ones(z_values.shape, dtype=int)*iterations
    exceeded_limit_before = np.zeros(z_values.shape, dtype=bool)
    for i in range(iterations):
        exceeded_limit_after = np.abs(z_values) >= 2
        iteration_counter[exceeded_limit_after & ~exceeded_limit_before] = i
        z_values[~exceeded_limit_after] = et_function(z_values[~exceeded_limit_after], plane[~exceeded_limit_after],
                                                      *et_f_args, **et_f_kwargs)
        exceeded_limit_before = exceeded_limit_after
    return iteration_counter, z_values


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
def max_var_segment_diff(plane: np.ndarray, zoom_factor: float, iterations: np.int64) -> np.ndarray:
    a = np.empty(2)
    np.round(np.array(plane.shape) / zoom_factor, 0, a)
    x_factor, y_factor = a.astype(np.int64)
    variance = 0
    segment_index_range = np.ones((2, 2)) * -1
    # TODO: put np.diff outside of loop for faster computation.
    #vertical_diff = np.diff(plane, axis=0)
    #horizontal_diff = np.diff(plane, axis=1)

    for x in range(plane.shape[0] - x_factor):
        for y in range(plane.shape[1] - y_factor):
            segment = plane[x:x + x_factor + 1, y:y + y_factor + 1]
            segment_variance = new_variance(segment.astype(np.int64), iterations)
            if segment_variance > variance:
                variance = segment_variance
                segment_index_range[:] = np.array([[x, x + x_factor], [y, y + y_factor]])

    return segment_index_range.astype(np.int64)


@njit
def max_var_segment(plane: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Identifies the segment of a 2D array with the largest variance.

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
    The variance measures how "spread out" the values in the array are.
    Variance is calculated as :math:`V = \frac{1}{n}\sum_{i=1}^n (x_i-\bar x)^2`.
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


if __name__ == '__main__':
    colormap = 'twilight'
    re_lim = (-2., 2.)
    im_lim = (-1.5, 1.5)
    resolution = 500
    iterations = 100

    et_function = lambda z, c: (np.abs(z.real) + 1j * np.abs(z.imag)) ** 2 + c
    complex_plane = np.linspace(*re_lim, resolution) + 1j * np.linspace(*im_lim, resolution)[:, np.newaxis]
    iter_count, z_values = generate_escape_time(complex_plane, iterations, et_function)
    z_abs = np.abs(z_values)

    plt.ioff()
    fig, axes = plt.subplots(ncols=2, nrows=2)
    zoomout_axes = axes[0]
    zoomin_axes = axes[1]
    zoomout_axes[0].imshow(iter_count, cmap=colormap, extent=re_lim + im_lim)
    zoomout_axes[1].imshow(z_abs, cmap=colormap, extent=re_lim + im_lim)

    x, y = max_var_segment(iter_count, 5)
    iter_count_zoom = iter_count[x[0]:x[1], y[0]:y[1]]
    zoomin_axes[0].imshow(iter_count_zoom, cmap=colormap,
                          extent=(complex_plane[x[0], y[0]].real, complex_plane[x[1], y[1]].real,
                                  complex_plane[x[0], y[0]].imag, complex_plane[x[1], y[1]].imag))

    x, y = max_var_segment(z_abs, 5)
    z_abs_zoom = z_abs[x[0]:x[1], y[0]:y[1]]
    zoomin_axes[1].imshow(z_abs_zoom, cmap=colormap,
                          extent=(complex_plane[x[0], y[0]].real, complex_plane[x[1], y[1]].real,
                                  complex_plane[x[0], y[0]].imag, complex_plane[x[1], y[1]].imag))
    plt.show()
