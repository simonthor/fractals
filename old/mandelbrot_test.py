import numpy as np
import matplotlib.pyplot as plt


def calc_mandelbrot(c, maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z**2 + c
    return maxiter


def main():
    maxiter = 100
    res = 1000
    n3 = np.zeros((res, res))
    x = np.linspace(-2, 2, res)
    y = np.linspace(-2, 2, res)

    for xcol in range(res):
        for ycol in range(res):
            n = calc_mandelbrot(x[xcol] + 1j*y[ycol], maxiter)
            n3[ycol, xcol] = n

    plt.imshow(n3)
    plt.savefig("mandelbrot.png")
