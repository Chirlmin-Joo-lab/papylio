import numpy as np #scientific computing with Python
import xarray as xr
import matplotlib.pyplot as plt

# import matplotlib.patches as patches
# from matplotlib.collections import PatchCollection
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# from papylio.molecule import Molecule


def marginal_hist2d(x, y, bins=10, range=(0, 1), xlabel=None, ylabel=None, count_label='Count', show_marginal=True,
                    show_colorbar=False, ax=None, **hist2d_kwargs):
    """
    Generates a 2D histogram plot with optional marginal histograms and colorbar.

    Parameters:
    -----------
    x : array-like or xarray.DataArray
        The data for the x-axis. If an xarray.DataArray is provided, its name will be used as the x-axis label.
    y : array-like or xarray.DataArray
        The data for the y-axis. If an xarray.DataArray is provided, its name will be used as the y-axis label.
    bins : int or tuple of int, optional (default=100)
        Number of bins for the 2D histogram. If an integer is provided, it will be used for both x and y axes.
    range : tuple or tuple of two tuples, optional (default=((0, 1), (0, 5000)))
        The range of the data for the histogram in the form (min, max) or ((xmin, xmax), (ymin, ymax)).
    ax : matplotlib.axes.Axes, optional (default=None)
        The axes object to plot on. If None, a new figure and axes will be created.
    xlabel : str, optional (default=None)
        Label for the x-axis. If not provided, the name of the `x` data will be used (if available).
    ylabel : str, optional (default=None)
        Label for the y-axis. If not provided, the name of the `y` data will be used (if available).
    show_marginal : bool, optional (default=True)
        Whether to show the marginal histograms along the x and y axes.
    show_colorbar : bool, optional (default=True)
        Whether to display a colorbar next to the 2D histogram.
    hist2d_kwargs : dict, optional (default=None)
        Additional keyword arguments passed to `matplotlib.pyplot.hist2d` for customizing the 2D histogram.
    count_label : str, optional (default='Counts')
        Label for the count axis (used in the marginal histograms and colorbar).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    axs : list of matplotlib.axes.Axes
        A list containing the axes objects for the main plot, marginal histograms (if shown), and colorbar (if shown).

    """

    if ax is None:
        fig, ax_main = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
        ax_main = ax

    if isinstance(x, xr.DataArray):
        if xlabel is None:
            xlabel = x.name
        x = x.values.flatten()

    if isinstance(y, xr.DataArray):
        if ylabel is None:
            ylabel = y.name
        y = y.values.flatten()

    if isinstance(bins, int):
        bins = [bins] * 2

    if np.issubdtype(type(range[0]), np.number):
        range = (range,) * 2

    # Main 2D histogram
    h = ax_main.hist2d(x, y, bins=bins, range=range, **hist2d_kwargs)
    ax_main.set_box_aspect(1)

    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)

    axs = [ax_main]

    # Marginal histograms
    if show_marginal == True:
        cmap = plt.get_cmap('viridis')
        marginal_hist_kwargs = dict(fc=cmap(0))

        spacing = 0.01
        xyhist_size = 0.15
        bbox_main = ax_main.get_position()

        ax_xhist = fig.add_axes(
            rect=[bbox_main.x0, bbox_main.y0 + bbox_main.height + spacing, bbox_main.width, xyhist_size],
            sharex=ax_main)
        ax_yhist = fig.add_axes(
            rect=[bbox_main.x0 + bbox_main.width + spacing, bbox_main.y0, xyhist_size, bbox_main.height],
            sharey=ax_main)

        ax_xhist.hist(x, bins=bins[0], range=range[0], **marginal_hist_kwargs)
        ax_yhist.hist(y, bins=bins[1], range=range[1], orientation='horizontal', **marginal_hist_kwargs)

        # Hide the tick labels on the marginal histograms
        ax_xhist.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_yhist.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        # Set custom ticks excluding 0
        x_hist_y_ticks = ax_xhist.get_yticks()
        y_hist_x_ticks = ax_yhist.get_xticks()
        ax_xhist.set_yticks(x_hist_y_ticks[1:])
        ax_yhist.set_xticks(y_hist_x_ticks[1:])

        # Set labels
        ax_xhist.set_ylabel(count_label)
        ax_yhist.set_xlabel(count_label)

        axs.append(ax_xhist)
        axs.append(ax_yhist)

    # Create a colorbar axis that matches the height of the main plot
    if show_colorbar:
        bbox_yhist = ax_yhist.get_position()

        ax_cbar = fig.add_axes(
            rect=[bbox_main.x0 + bbox_main.width + bbox_yhist.width + spacing * 2, bbox_main.y0, xyhist_size * 0.2,
                  bbox_main.height])
        fig.colorbar(h[3], cax=ax_cbar)
        ax_cbar.set_ylabel(count_label)

        axs.append(ax_cbar)

    return fig, axs


def histogram(da, axis=None, **hist_kwargs):
    if axis is None:
        figure, axis = plt.subplots()
    else:
        figure = axis.figure

    if 'channel' in da.dims:
        das = [da.sel(channel=channel) for channel in da.channel]
    else:
        das = [da]

    if len(das) > 1:
        hist_kwargs['histtype'] = 'step'

    for da in das:
        da.plot.hist(ax=axis, **hist_kwargs)
    axis.set_ylabel('Count')
    axis.set_title('')

    # if save:
    #     fig.savefig(self.absoluteFilePath.with_name(f'{self.name}_{parameter}_histogram').with_suffix('.png'))

    return figure, axis



def histogram_FRET(data, axis, **kwargs):
    axis.hist(data, range=(0, 1), **kwargs)
    axis.set_xlim((0, 1))
    axis.set_xlabel('FRET')
    axis.set_ylabel('Count')

def fit_hist(data, axis):
    hist, bin_edges = np.histogram(data, 100, range=(0, 1))
    bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2

    # plt.plot(bin_centers,hist)

    from scipy.signal import butter
    from scipy.signal import filtfilt
    b, a = butter(2, 0.2, 'low')
    output_signal = filtfilt(b, a, hist)
    plt.plot(bin_centers, output_signal)

    from scipy.signal import find_peaks
    peaks, properties = find_peaks(output_signal, prominence=5, width=7)  # prominence=1
    plt.plot(bin_centers[peaks], hist[peaks], "x")

    def func(x, a, b, c, d, e, f):
        return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d * np.exp(-(x - e) ** 2 / (2 * f ** 2))

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(func, bin_centers, hist, method='trf',
                           p0=[hist[peaks[0]], bin_centers[peaks[0]], 0.1, hist[peaks[1]], bin_centers[peaks[1]], 0.1],
                           bounds=(0, [np.inf, 1, 1, np.inf, 1, 1]))

    axis.plot(bin_centers, func(bin_centers, *popt))
    # plt.plot(bin_centers,func(bin_centers, 10000,0.18,0.1,5000,0.5,0.2))

# uniqueFileNames = list(set([re.search('hel[0-9]*',fileName).group() for fileName in fileNames]))


def show_image_3d(image, figure=None):
    if not figure:
        figure = plt.figure()

    from matplotlib import cm
    axis = figure.gca(projection='3d')
    X = np.arange(image.shape[1])
    Y = np.arange(image.shape[0])
    X, Y = np.meshgrid(X, Y)
    axis.plot_surface(X, Y, image, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

