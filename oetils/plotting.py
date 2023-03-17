from math import sqrt

import matplotlib.pyplot as plt
import numpy as np


def init_plotting(latex=None, publish=False, W=None, pad=None, beamer=False,
        poster=False, seaborn=False, serif=False, show=False):
    plt.plot()    # Needs to be done for some reason
    plt.close()
    if latex is None:
        latex = publish or beamer or poster
    if W is None:
        W = 5.8 if publish else 6.3 if beamer else 8.27
    if pad is None:
        pad = 0.01 if publish or beamer or poster else 0.2

    W -= 2*pad
    rc = {
        'text.usetex': latex,
        'figure.constrained_layout.use': True,
        'figure.dpi': 120 if show else 300,
        'figure.figsize': (W, W/sqrt(2)),
        'savefig.pad_inches': pad,
        'savefig.bbox': 'tight',
        'savefig.dpi': 120 if show else 300
    }
    if seaborn:
        import seaborn as sns
        sns.set()
    else:
        plt.style.use(('science', 'grid'))
    if publish:
        rc.update({
            'font.size': 8,
            'axes.labelsize': 8,
            'axes.titlesize': 8,
            'legend.fontsize': 8,
            'legend.title_fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'grid.alpha': 0.2,
            'font.family': 'lmodern'
        })
    elif beamer:
        rc.update({
            'font.size' : 11,
            'axes.labelsize': 11,
            'legend.fontsize': 11,
            'grid.alpha': 0.2,
            'font.family': 'sans-serif',
        })
    elif poster:
        # Still using A5 width (ca. A0 width / 6) because very large figures look weird, everything
        # becomes too thin. This way the text is readable from a but further as well.
        rc.update({
            'font.size' : 14,
            'axes.labelsize': 14,
            'legend.fontsize': 14,
            'grid.alpha': 0.2,
            'font.family': 'sans-serif',
        })
    else:
        rc.update({
            'font.size' : 11,
            'axes.labelsize': 11,
            'legend.fontsize': 11,
            'grid.alpha': 0.2,
            'font.family': 'sans-serif'
        })
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
            ) if serif else '')
        })

    plt.rcParams.update(rc)
    return W


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
