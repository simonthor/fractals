import numpy as np
from matplotlib import pyplot as plt


def generate_mandelbrot(plane: np.ndarray, iterations: int):
    """Generates the mandelbrot set.

    Parameters
    ----------
    plane : numpy.ndarray
        The coordinates in the complex plane that will be used to generate the mandelbrot set.
    iterations : int
        Maximum number of iterations that the mandelbrot formula will be used on a given number.

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
        z_values[~exceeded_limit_after] = z_values[~exceeded_limit_after]**2 + plane[~exceeded_limit_after]
        exceeded_limit_before = exceeded_limit_after
    return iteration_counter, z_values


if __name__ == '__main__':
    re_lim = (-2., 1.)
    im_lim = (-1.5, 1.5)
    resolution = 1000
    iterations = 100
    complex_plane = np.linspace(*re_lim, resolution) + 1j*np.linspace(*im_lim, resolution).reshape((resolution, 1))
    fractal_image, z_values = generate_mandelbrot(complex_plane, iterations)
    fig, axes = plt.subplots(ncols=2, sharex='all', sharey='all')
    axes[0].imshow(fractal_image, cmap='jet', extent=re_lim+im_lim)
    axes[1].imshow(np.abs(z_values), cmap='jet', extent=re_lim+im_lim)
    axes[0].set_xlabel('$Re(z)$')
    axes[0].set_ylabel('$Im(z)$')
    plt.show()
