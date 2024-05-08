from copy import deepcopy
from math import sqrt
from tempfile import NamedTemporaryFile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa
from scipy.optimize import root_scalar
from tueplots import bundles


def init_plotting(
        venue=None, latex=None, W=None, pad=None, sans=False, sansmath=False, show=False, dots=3000,
        bundle_kwargs=None, **rcparams):
    plt.plot()    # Needs to be done for some reason
    plt.close()
    latex = latex or bool(venue)
    bundle = bundles.__dict__[venue](**(bundle_kwargs or {})) if venue in dir(bundles) else None
    if W is None:
        W = 8.27
        if venue == 'paper':
            W = 5.8
        elif venue == 'beamer':
            W = 6.3
        elif bundle:
            W = bundle['figure.figsize'][0]
    if pad is None:
        pad = 0.01 if bool(venue) else 0.2

    H = W / sqrt(2) - 2*pad + 2*plt.rcParams['figure.constrained_layout.h_pad']
    W = W - 2*pad + 2*plt.rcParams['figure.constrained_layout.w_pad']
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use(('science', 'grid'))
    rc = {
        'text.usetex': latex,
        'figure.constrained_layout.use': True,
        'figure.dpi': 120 if show else dots / W,
        'figure.figsize': (W, H),
        'savefig.bbox': 'tight',
        'savefig.pad_inches': pad,
        'savefig.dpi': 120 if show else dots / W,
        'grid.linestyle': ':',
        'grid.alpha': 0.2,
        'legend.fancybox': False,
        'legend.framealpha': 0.8,
        'legend.edgecolor': 'inherit',
        'patch.linewidth': 0.5,
        'lines.linewidth': 1,
        'errorbar.capsize': 2,
        'font.size': 11,
        'axes.labelsize': 11,
        'legend.fontsize': 11,
        'font.family': 'sans-serif' if sans or sansmath else 'serif',
    }
    if latex:
        rc.update({
            'text.latex.preamble': (
                r'\usepackage{lmodern}'
                r'\usepackage[T1]{fontenc}'
                r'\usepackage[utf8]{inputenc}'
                r'\usepackage{amssymb}'
                r'\usepackage{amsmath}'
                r'\usepackage{siunitx}'
                r'\usepackage{physics}'
                r'\usepackage{bm}'
            ) + ((
                r'\usepackage{sansmath}'
                r'\sansmath'
            ) if sansmath else '')
            + (bundle['text.latex.preamble'] if bundle else '')
        })
    if venue == 'paper' or bundle:
        rc.update({
            'axes.labelsize': (bundle['axes.labelsize'] if bundle else 8),
            'axes.titlesize': (bundle['axes.titlesize'] if bundle else 8),
            'font.size': (bundle['font.size'] if bundle else 8),
            'legend.fontsize': (bundle['legend.fontsize'] if bundle else 6),
            'xtick.labelsize': (bundle['xtick.labelsize'] if bundle else 6),
            'ytick.labelsize': (bundle['ytick.labelsize'] if bundle else 6),
            'axes.linewidth': 0.4,
            'errorbar.capsize': 1,
            'grid.linewidth': 0.3,
            'legend.title_fontsize': 8,
            'lines.linewidth': 0.8,
            'lines.markeredgewidth': 0.8,
            'lines.markersize': 3,
            'patch.linewidth': 0.4,
            'xtick.major.width': 0.3,
            'xtick.major.size': 2,
            'xtick.minor.width': 0.3,
            'xtick.minor.size': 1,
            'ytick.major.width': 0.3,
            'ytick.major.size': 2,
            'ytick.minor.width': 0.3,
            'ytick.minor.size': 1,
        })
    elif venue == 'poster':
        rc.update({
            'font.size': 14,
            'axes.labelsize': 14,
            'legend.fontsize': 14,
        })
    plt.rcParams.update(rc | rcparams)
    return W


def savefig(fig, path, tries=20, width=None, height=None, pad=None, v=True):
    """Save figure with true size."""
    fig_ = deepcopy(fig)
    pad = pad or plt.rcParams['savefig.pad_inches']
    w_pad = 2 * (pad - plt.rcParams['figure.constrained_layout.w_pad'])
    h_pad = 2 * (pad - plt.rcParams['figure.constrained_layout.h_pad'])
    dpi = fig_.get_dpi()
    target_width = width = width or fig_.get_figwidth() + w_pad
    target_height = height = height or fig_.get_figheight() + h_pad

    def size(w, h):
        fig_.set_size_inches([w - w_pad, h - h_pad])
        with NamedTemporaryFile(suffix='.png') as f:
            fig_.savefig(f.name, pad_inches=pad)
            h, w, _ = plt.imread(f.name).shape
            w, h = w / dpi, h / dpi
        return w, h

    def w_error(x):
        w, _ = size(x, height)
        return target_width - w

    def h_error(x):
        _, h = size(width, x)
        return target_height - h

    if v: print('Computing figsize...', end=' ')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        width = root_scalar(w_error, x0=width, x1=1.1*width, maxiter=tries).root
        if v: print(f'width error: {abs(w_error(width)):f} (original: {abs(w_error(target_width)):f})', end=', ')
        height = root_scalar(h_error, x0=height, x1=1.1*height, maxiter=tries).root
        if v: print(f'height error: {abs(h_error(height)):f} (original: {abs(h_error(target_height)):f})')
    plt.close(fig_)
    fig.set_size_inches([width - w_pad, height - h_pad])
    fig.savefig(path, pad_inches=pad)


class LivePlot:
    """From here: https://matplotlib.org/stable/tutorials/advanced/blitting.html.
    Modified slightly (added a plt.pause call) to support macOS backend.
    """
    def __init__(self, canvas, animated_artists=(), pause=0.01):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []
        self.pause = pause

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()
        plt.pause(self.pause)


if __name__ == "__main__":
    # Init plotting
    init_plotting(show=True)

    # Test live plotting
    fig, ax = plt.subplots()
    x = np.linspace(0, 2 * np.pi, 100)

    # add a line
    (ln,) = ax.plot(x, np.sin(x), animated=True)

    # add a frame number
    fr_number = ax.annotate(
        "0",
        (0, 0),
        xycoords="axes fraction",
        xytext=(10, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        animated=True,
    )
    bm = LivePlot(fig.canvas, [ln, fr_number])

    # make sure our window is on the screen and drawn
    plt.show(block=False)
    plt.pause(.1)

    for j in range(1000):
        # update the artists
        ln.set_ydata(np.sin(x + (j / 100) * np.pi))
        fr_number.set_text("frame: {j}".format(j=j))
        # tell the blitting manager to do its thing
        bm.update()
