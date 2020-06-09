import numpy as np
from matplotlib import pyplot as plt
from typing import Callable, Tuple
import sys


def generate_escape_time(re_lim: Tuple[float, float], im_lim: Tuple[float, float], iterations: int, resolution: int,
                         et_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                         *et_f_args, **et_f_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Generates an escape time fractal.

    Parameters
    ----------
    complex_plane : numpy.ndarray
        The coordinates in the complex complex_plane that will be used to generate the mandelbrot set.
    iterations : int
        Maximum number of iterations that the mandelbrot formula will be used on a given number.
    et_function : function
        The function used to calculate the z value of each point in the complex complex_plane.
        Must at least accept (but do not have to use) 2 input arguments of the complex complex_plane and the current z values.
    *et_f_args : any, optional
        Other input arguments for et_function.
    **et_f_kwargs : any, optional
        Other input keyword arguments for et_function.

    Returns
    ----------
    iteration_counter : numpy.ndarray
        Number of iterations before |z| >= 2. Same shape as complex_plane.
    z_values : numpy.ndarray
        The values of the complex numbers right after |z| >= 2. Same shape as complex_plane.
    """
    complex_plane = np.linspace(*re_lim, resolution) + 1j * np.linspace(*im_lim, resolution)[:, np.newaxis]
    z_values = np.copy(complex_plane)
    iteration_counter = np.ones(z_values.shape, dtype=int)*iterations
    exceeded_limit_before = np.zeros(z_values.shape, dtype=bool)
    for i in range(iterations):
        exceeded_limit_after = np.abs(z_values) >= 2
        iteration_counter[exceeded_limit_after & ~exceeded_limit_before] = i
        z_values[~exceeded_limit_after] = et_function(z_values[~exceeded_limit_after], complex_plane[~exceeded_limit_after],
                                                      *et_f_args, **et_f_kwargs)
        exceeded_limit_before = exceeded_limit_after
    return iteration_counter, z_values


if __name__ == '__main__':
    colormap = sys.argv[1]
    re_lim = (-2., 2.)
    im_lim = (-1.5, 1.5)
    resolution = 500
    iterations = 100

    et_function = lambda z, c: (z-c)**1.5
    complex_plane = np.linspace(*re_lim, resolution) + 1j * np.linspace(*im_lim, resolution)[:, np.newaxis]
    iter_count, z_values = generate_escape_time(complex_plane, iterations, et_function)
    z_abs = np.abs(z_values)

    fig, axes = plt.subplots(ncols=2, nrows=1)
    plot_kwargs = dict(cmap=colormap, extent=re_lim + im_lim, origin='lower')
    axes[0].imshow(iter_count, **plot_kwargs)
    axes[1].imshow(z_abs, **plot_kwargs)
    fig.tight_layout()
    plt.show()
