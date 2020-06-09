import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from fractal_generator import *
from progress import *
from typing import Iterable


# TODO: change re_lim and im_lim to plane_range for better support for factory_args
def animate(filename: str, frames: int, colormap: str, iter_args: dict = {}, const_args: dict = {}, factory_args: dict = {},
            graph_type: str = 'i', dpi: int = 300, anim_kwargs: dict = {'fps': 24}):

    for key, val in iter_args.items():
        if not isinstance(val, Iterable):
            raise TypeError(f'{key} must be iterable but got type {type(val)}.')

    plt.ioff()
    fig, ax = plt.subplots()
    image = ax.imshow(np.zeros((2, 2)), cmap=colormap, origin='lower')
    animation_writer = animation.FFMpegWriter(**anim_kwargs, extra_args=['-vcodec', 'libx264'])

    with animation_writer.saving(fig, filename, dpi=dpi):
        for i, parameter_values in enumerate(zip(*iter_args.values())):
            one_time_args = dict(zip(iter_args.keys(), parameter_values))
            total_args = {**one_time_args, **const_args}
            if i > 0:
               for key, val in f_onetime_args.items():
                   total_args[key] = val

            iter_count, z_values = generate_escape_time(**total_args)
            if graph_type == 'i':
                image.set_data(iter_count)
            elif graph_type == 'z':
                image.set_data(np.abs(z_values))
            else:
                raise ValueError(f"'{graph_type}' is not a valid value for graph_type; supported values are 'i', 'z'")

            image.set_extent(total_args['re_lim'] + total_args['im_lim'])
            image.autoscale()

            animation_writer.grab_frame()
            print_progressbar(i + 1, frames, 'rendering:')

            f_onetime_args = {key: f(iter_count, z_values) for key, f in factory_args.items()}


def snapshot(re_lim: Tuple[float, float], im_lim: Tuple[float, float], iterations: int, resolution: int,
             et_function: Callable[[np.ndarray, np.ndarray], np.ndarray], filename: str, colormap: str,
             graph_type: str = 'i', dpi: int = 300, *et_f_args, **et_f_kwargs):
    iter_count, z_values = generate_escape_time(re_lim, im_lim, iterations, resolution, et_function, *et_f_args, **et_f_kwargs)

    if graph_type == 'i':
        plt.imshow(iter_count, cmap=colormap, extent=re_lim + im_lim, origin='lower')
    elif graph_type == 'z':
        plt.imshow(z_values, cmap=colormap, extent=re_lim + im_lim, origin='lower')

    plt.savefig(filename, dpi=dpi)
