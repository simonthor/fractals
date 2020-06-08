import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from interesting_region import *
from fractal_generator import *

if __name__ == "__main__":
    colormap = 'inferno'
    re_lim = (-2., 2.)
    im_lim = (-1.8, 0.2)
    resolution = 300
    iterations = 60
    frames = 50
    zoom_factor = 1.1
    video_filename = './burning_ship.mp4'

    et_function = lambda z, c: (np.abs(z.real) + 1j * np.abs(z.imag)) ** 2 + c

    plt.ioff()
    fig, ax = plt.subplots()
    image = ax.imshow(np.zeros((2, 2)), cmap=colormap, origin='lower')
    animation_writer = animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])

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


# def animate(data, index):
#     """Function that animates all particles in 3D.
#     NOTE: This function is rather slow. Improvements might be possible.
#     Returns: None but creates an mp4 file with the animation."""
#
#     frames = []
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#
#     # Settings
#     boxsize = np.nanmax(data[np.isfinite(data)])
#     ax.set_xlim(0, boxsize)
#     ax.set_ylim(0, boxsize)
#     ax.set_zlim(0, boxsize)
#
#     ax.set_axis_off()
#     fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
#     fig.set_size_inches((5, 5), forward=True)
#
#     color = np.arange(data.shape[1])
#
#     if args['save_video']:
#         writer = animation.FFMpegWriter(fps=args['fps'], bitrate=args['fps']*100, extra_args=['-vcodec', 'libx264', '-preset', args['preset']])
#         tbefore = time.time()
#         bcoord = np.zeros(data.shape[1])
#         plot = ax.scatter(bcoord, bcoord, bcoord, s=args['particlesize'], c=color, cmap="jet")
#         nframes = data.shape[0]
#
#         with writer.saving(fig, f"./videos/{index if index >= 0 else ''}{args['videoname']}.mp4", dpi=args['dpi']):
#             for i, frame in enumerate(data):
#                 plot._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])
#                 # grab_frame takes time
#                 writer.grab_frame()
#
#         tafter = time.time()
#         print("time taken to save video:", tafter - tbefore, "s")
#     else:
#         video = []
#         for frame in data:
#             plot = ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], s=args['particlesize'], c=np.arange(data.shape[1]), cmap="jet", animated=True)
#             video.append([plot])
#
#         animation.ArtistAnimation(fig, video, interval=1000/args['fps'], blit=True, repeat_delay=1000)
#         plt.show()