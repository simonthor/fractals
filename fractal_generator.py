import numpy as np
from matplotlib import pyplot as plt
from typing import Callable, Tuple


def generate_escape_time(plane: np.ndarray, iterations: int, et_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
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
        Must at least accept two input arguments of the complex plane and the current z values.
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
    iteration_counter = np.zeros(z_values.shape, dtype=int)
    exceeded_limit_before = np.zeros(z_values.shape, dtype=bool)
    for i in range(iterations):
        exceeded_limit_after = np.abs(z_values) >= 2
        iteration_counter[exceeded_limit_after & ~exceeded_limit_before] = i
        z_values[~exceeded_limit_after] = et_function(z_values[~exceeded_limit_after], plane[~exceeded_limit_after],
                                                      *et_f_args, **et_f_kwargs)
        exceeded_limit_before = exceeded_limit_after
    return iteration_counter, z_values


if __name__ == '__main__':
    colormap = 'twilight'
    re_lim = (-2., 2.)
    im_lim = (-1.5, 1.5)
    resolution = 500
    iterations = 100
    et_function = lambda z, c: (np.abs(np.real(z)) + 1j*np.abs(np.imag(z)))**2 + c

    complex_plane = np.linspace(*re_lim, resolution) + 1j*np.linspace(*im_lim, resolution)[:, np.newaxis]
    fractal_image, z_values = generate_escape_time(complex_plane, iterations, et_function)
    fig, axes = plt.subplots(ncols=2, sharex='all', sharey='all')
    axes[0].imshow(fractal_image, cmap=colormap, extent=re_lim+im_lim)
    axes[1].imshow(np.abs(z_values), cmap=colormap, extent=re_lim+im_lim)
    axes[0].set_xlabel('$Re(z)$')
    axes[0].set_ylabel('$Im(z)$')
    plt.show()
