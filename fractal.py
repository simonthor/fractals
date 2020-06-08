from dataclasses import dataclass, field
from typing import Callable, Tuple, Sequence, Dict
import numpy as np
from matplotlib import pyplot as plt


@dataclass
class EscapeTimeFractal:
    re_limit: Tuple[float, float]
    im_limit: Tuple[float, float]
    resolution: int
    iterations: int
    et_function: Callable#[[np.ndarray, np.ndarray], np.ndarray]

    complex_plane: np.ndarray = field(init=False)
    z_values: np.ndarray = field(init=False)
    iteration_counts: np.ndarray = field(init=False)

    #calculate_plane: bool = False
    et_f_args: Sequence = field(default_factory=list)
    et_f_kwargs: Dict = field(default_factory=dict)

    def __post_init__(self):
    #   if self.calculate_plane:
        self.complex_plane = np.linspace(*self.re_limit, self.resolution) + 1j * np.linspace(*self.im_limit, self.resolution)[:, np.newaxis]

    def generate_escape_time(self, et_f_args, et_f_kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generates the escape time fractal.

        Parameters
        ----------
        *et_f_args : any, optional
            Other input arguments for et_function.
        **et_f_kwargs : any, optional
            Other input keyword arguments for et_function.

        Returns
        ----------
        iteration_counts : numpy.ndarray
            Number of iterations before |z| >= 2. Same shape as plane.
        z_values : numpy.ndarray
            The values of the complex numbers right after |z| >= 2. Same shape as plane.
        """

        z_values = np.copy(self.complex_plane)
        iteration_counts = np.zeros(z_values.shape, dtype=int)
        exceeded_limit_before = np.zeros(z_values.shape, dtype=bool)
        for i in range(self.iterations):
            exceeded_limit_after = np.abs(z_values) >= 2
            iteration_counts[exceeded_limit_after & ~exceeded_limit_before] = i
            z_values[~exceeded_limit_after] = self.et_function(z_values[~exceeded_limit_after],
                                                               self.complex_plane[~exceeded_limit_after],
                                                               *et_f_args, **et_f_kwargs)
            exceeded_limit_before = exceeded_limit_after
        self.iteration_counts = iteration_counts
        self.z_values = z_values

        return iteration_counts, z_values

    # TODO: change this to be able to automate any parameter? Probably results in lower efficiency but more freedom.
    def animate(self, parameter: str, *args, **kwargs):
        animation_function = getattr(self, f'_{parameter}')
        return animation_function(*args, **kwargs)

    def _zoom(self):
        pass

    def _change_et_function(self):
        pass


    def plot(self, plotfig=True, savefig=False, image_name='', *plot_args, **plot_kwargs):
        """Plots the escape time fractal.

        Parameters
        ----------
        plotfig : bool, optional
            Whether to plot the image of the fractal or not. Default is True.
        savefig : bool, optional
            Whether to save the image of the fractal as an image or not. Deafult is False.
        image_name : str, optional
            Name of the image of the fractal. Only used if savefig = True.
        *plot_args : any, optional
            Input arguments for ax.imshow.
        **plot_kwargs : any, optional
            Input keyword arguments for ax.imshow.

        Returns
        ----------
        fig : matplotlib.figure.Figure
            The figure of the plot.
        axes : matplotlib.axes.Axes
            The axes (can be multiple) with the plots.
        """

        fig, axes = plt.subplots(ncols=2, sharex='all', sharey='all')
        axes[0].imshow(self.iteration_counts, extent=self.re_limit + self.im_limit)
        axes[1].imshow(np.abs(self.z_values), extent=self.re_limit + self.im_limit)
        axes[0].set_xlabel('$Re(z)$')
        axes[0].set_ylabel('$Im(z)$')
        plt.show()

        return fig, axes
