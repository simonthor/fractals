import matplotlib.pyplot as plt
import numpy as np
from numba import jit

@jit
def calc_mandelbrot(c, maxiter):
    z = c
    for n in range(maxiter):
        if np.abs(z) > 2:
            return n
        z = z*z + c
    return maxiter

@jit
def calc_etfractal(xmin, xmax, ymin, ymax, res, maxiter):
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymax, ymin, res)
    n3 = np.empty((res, res))
    for xcol in range(res):
        for ycol in range(res):
            n = calc_mandelbrot(x[xcol] + 1j * y[ycol], maxiter)
            n3[ycol, xcol] = n
    return n3


res = 1000      # number of pixels on each row and column
maxiter = 100   # maximum number of iterations for each point on the complex plane
side = 2        # length of side (both x and y-axis)
cmap = 'jet'    # color map that chooses a color for each point depending on the number of times it iterates
n3 = calc_etfractal(-side, side, -side, side, res, maxiter)     # n3: array with values of each pixel on the complex plane
plt.imshow(n3, extent=[-side, side, -side, side], cmap=cmap, interpolation='none')      # plot the image using matplotlib
plt.show()
