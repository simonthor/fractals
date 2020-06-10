import render
import numpy as np
from numba import njit
from interesting_region import find_interesting_region

@njit
def et_function(z, c):
    return z**2+c


if __name__ == '__main__':
    render.auto_zoom((-2, 2), (-1.5, 1.5), 100, 400, et_function, 'zoom.mp4', 100, 1.01, 'twilight', graph_type='z')
