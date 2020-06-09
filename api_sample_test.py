import render
import numpy as np
from numba import njit


@njit
def et_function(z, c):
    return z**2+c


if __name__ == '__main__':
    render.animate('test_video.mp4', 50, 'twilight', graph_type='z',
                   const_args={'iterations': 60, 're_lim': (-2, 2), 'im_lim': (-2, 2), 'resolution': 200, 'et_function': et_function})
