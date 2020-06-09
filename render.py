import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from fractal_generator import *
from progress import *
from typing import Iterable


def animate(filename: str, frames: int, colormap: str, iter_args: dict, const_args: dict, dpi: int = 300, fps: int = 30):
    for key, val in iter_args.items():
        if not isinstance(val, Iterable):
            raise TypeError(f'{key} must be iterable but got type {type(val)}.')

    plt.ioff()
    fig, ax = plt.subplots()
    image = ax.imshow(np.zeros((2, 2)), cmap=colormap, origin='lower')
    animation_writer = animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])

    with animation_writer.saving(fig, filename, dpi=dpi):
        for i, parameter_values in enumerate(zip(*iter_args.values())):
            one_time_args = dict(zip(iter_args.keys(), parameter_values))
            total_args = {**one_time_args, **const_args}
            iter_count, z_values = generate_escape_time(**total_args)

            image.set_data(iter_count)
            image.set_extent(total_args['re_lim'] + total_args['im_lim'])
            image.autoscale()

            animation_writer.grab_frame()
            print_progressbar(i + 1, frames, 'video:')
