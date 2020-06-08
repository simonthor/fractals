import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from interesting_region import *
from fractal_generator import *
from progress import *


if __name__ == "__main__":
    colormap = 'inferno'
    re_lim = (-2., 2.)
    im_lim = (-1.8, 0.2)
    resolution = 500
    iterations = 60
    frames = 200
    zoom_factor = 1.01
    fps = 20
    video_filename = './burning_ship.mp4'

    et_function = lambda z, c: (np.abs(z.real) + 1j * np.abs(z.imag)) ** 2 + c

    plt.ioff()
    fig, ax = plt.subplots()
    image = ax.imshow(np.zeros((2, 2)), cmap=colormap, origin='lower')
    animation_writer = animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])

    with animation_writer.saving(fig, video_filename, dpi=300):
        for i in range(frames):
            # print(f'frame {i}')
            complex_plane = np.linspace(*re_lim, resolution) + 1j * np.linspace(*im_lim, resolution)[:, np.newaxis]
            iter_count, z_values = generate_escape_time(complex_plane, iterations, et_function)
            #z_abs = np.abs(z_values)

            image.set_data(iter_count)
            image.set_extent(re_lim + im_lim)
            image.autoscale()
            #image.set_clim(iter_count.min(), iter_count.max())
            animation_writer.grab_frame()
            print_progressbar(i + 1, frames, 'video:')

            if i < frames - 1:
                x_range, y_range = find_interesting_region(iter_count, zoom_factor, iterations)#, np.int64(iterations))
                re_lim = (complex_plane[x_range[0], y_range[0]].real, complex_plane[x_range[1], y_range[1]].real)
                im_lim = (complex_plane[x_range[0], y_range[0]].imag, complex_plane[x_range[1], y_range[1]].imag)
