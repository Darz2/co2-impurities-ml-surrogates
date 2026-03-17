#!/usr/bin/env python

import matplotlib.pyplot as plt
import scienceplots
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib as mpl
import numpy as np
import os

class PlotSettings:
    """
    A class to manage matplotlib plot settings and provide static methods for plot initialization
    and styling. Includes color palettes, fonts, sizes, and utility functions for creating
    publication-quality figures.
    """

    # ============== PLOT DIMENSIONS ==============
    PLOT_SIZE = (4, 3)
    RESOLUTION = 1200

    # ============== FONTS ==============
    GRAPHIC_FONT = 'Arial'
    MATH_FONT = 'dejavuserif'  # ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']

    # ============== SPINE & TICK SETTINGS ==============
    SPINE_WIDTH = 1.5
    TICK_WIDTH = 0.75
    TICK_LENGTH = 4
    MINOR_TICK_WIDTH = 0.5
    MINOR_TICK_LENGTH = 2

    # ============== FONT SIZES ==============
    TICK_LABELSIZE = 10
    LABEL_FONTSIZE = 14
    LEGEND_FONTSIZE = 8

    # ============== LEGEND SETTINGS ==============
    LEGEND_BOXWIDTH = 0.75
    BORDERAXESPAD = 0.6

    # ============== LINE & MARKER SETTINGS ==============
    LINEWIDTH = 1
    MARKERSIZE = 4
    MARKEREDGEWIDTH = 0.75
    CAPSIZE = 3
    LEGEND_LINEWIDTH = 1

    # ============== TRANSPARENCY ==============
    ALPHA = 0.5

    # ============== SUBPLOT SPACING ==============
    WSPACE = 0.3
    HSPACE = 0.3

    # ============== SCREEN PREVIEW DPI ==============
    SCREEN_DPI = 150

    # ============== SUBPLOT LABEL SETTINGS ==============
    SUBPLOT_LABEL_FONTSIZE = 14
    SUBPLOT_LABEL_FONTWEIGHT = 'bold'

    # ============== TEXTBOX SETTINGS ==============
    TEXTBOX_FONTSIZE = 9
    TEXTBOX_ALPHA = 0.85

    # ============== COLOR PALETTES ==============
    COLORS = ['#e41a1c', '#008000', '#377eb8', '#ff7f00', '#984ea3', '#a65628', '#f781bf']

    COLOR_10 = [
        '#008000',      # green
        '#ff7f00',      # orange
        '#984ea3',      # purple
        '#a65628',      # brown
        '#f781bf',      # pink
        "#05545a",      # yellow
        '#4daf4a',      # light green
        '#ffb300',      # gold
        '#1f78b4',      # teal-ish (not blue)
        '#b15928',      # dark brown
        '#999999',      # gray
        "#570F0F",      # dark brown
        '#377eb8',      # blue
        '#dede00',      # light yellow
        '#a6cee3',      # light blue
        '#fdbf6f',      # light orange
        '#cab2d6',      # light purple
        '#ffff99',      # light greenish-yellow
    ]

    COLOR_8 = [
        "#e41a1d",
        "#0b57f9",
        "#03fdf3",
        '#984ea3',
        '#377eb8'
    ]

    DUAL_COLORS = [
        ("#06fb0b", "#008000"),     # green (face, edge)
        ("#56abff", "#0000FF"),     # blue  (face, edge)
        ("#FF5656", "#FF0000"),     # red   (face, edge)
        ("#F68FC2", "#FF1493"),     # pink  (face, edge)
        ("#A6CEE3", "#1F78B4"),     # light blue (face, edge)
    ]

    MARKERS = ['o', 's', '^', 'D', 'h', 'v', 'p', '*', 'X', '<', '>', '8', 'P', '|', '_']

    @staticmethod
    def _initialize_latex():
        """Initialize LaTeX rendering in matplotlib."""
        mpl.rcParams['text.usetex'] = True

    @staticmethod
    def get_face_colors(colors=None, alpha=None):
        """
        Convert color specifications to RGBA tuples with specified alpha.

        Parameters:
        -----------
        colors : list, optional
            List of color specifications (hex or named colors).
            If None, uses PlotSettings.COLORS
        alpha : float, optional
            Alpha transparency value (0-1).
            If None, uses PlotSettings.ALPHA

        Returns:
        --------
        list : List of RGBA tuples
        """
        if colors is None:
            colors = PlotSettings.COLORS
        if alpha is None:
            alpha = PlotSettings.ALPHA

        rgba_colors = [mcolors.to_rgba(c) for c in colors]
        return [(rgba[0], rgba[1], rgba[2], alpha) for rgba in rgba_colors]

    @staticmethod
    def plot_init(figsize=None, style='ieee'):
        """
        Create a matplotlib figure and axis with predefined styles.

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size as (width, height).
            If None, uses PlotSettings.PLOT_SIZE
        style : str, optional
            Matplotlib style context to apply.
            Default is 'ieee'

        Returns:
        --------
        tuple : (fig, ax)
            Matplotlib figure and axes objects
        """
        if figsize is None:
            figsize = PlotSettings.PLOT_SIZE

        PlotSettings._initialize_latex()

        with plt.style.context([style]):
            plt.rcParams['font.family'] = PlotSettings.GRAPHIC_FONT
            plt.rcParams['mathtext.fontset'] = PlotSettings.MATH_FONT
            plt.rcParams['text.usetex'] = True

            fig, ax = plt.subplots(figsize=figsize)

            # Set spine widths
            for spine in ax.spines.values():
                spine.set_linewidth(PlotSettings.SPINE_WIDTH)

            # Apply tick parameters
            ax.tick_params(
                axis='both', which='major', direction='in',
                width=PlotSettings.TICK_WIDTH, length=PlotSettings.TICK_LENGTH,
                labelsize=PlotSettings.TICK_LABELSIZE,
                bottom=True, top=True, left=True, right=True
            )
            ax.tick_params(
                axis='both', which='minor', direction='in',
                width=PlotSettings.MINOR_TICK_WIDTH, length=PlotSettings.MINOR_TICK_LENGTH,
                bottom=True, top=True, left=True, right=True
            )

            return fig, ax

    @staticmethod
    def style_legend(ax, loc="upper right", ncol=1, edgecolor="black",
                     frame=True, fontsize=None, boxwidth=None, **kwargs):
        """
        Style a legend on an axes object with predefined settings.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes object to add legend to
        loc : str, optional
            Legend location (default: 'upper right')
        ncol : int, optional
            Number of columns (default: 1)
        edgecolor : str, optional
            Edge color of legend frame (default: 'black')
        frame : bool, optional
            Whether to show legend frame (default: True)
        fontsize : int, optional
            Font size for legend text.
            If None, uses PlotSettings.LEGEND_FONTSIZE
        boxwidth : float, optional
            Line width of legend box.
            If None, uses PlotSettings.LEGEND_BOXWIDTH
        **kwargs : dict
            Additional arguments passed to ax.legend()

        Returns:
        --------
        matplotlib.legend.Legend
            The legend object
        """
        if fontsize is None:
            fontsize = PlotSettings.LEGEND_FONTSIZE
        if boxwidth is None:
            boxwidth = PlotSettings.LEGEND_BOXWIDTH

        # Coerce ncol to int
        try:
            ncol = int(ncol)
        except (ValueError, TypeError):
            ncol = 1

        if "ncol" in kwargs:
            try:
                kwargs["ncol"] = int(kwargs["ncol"])
            except (ValueError, TypeError):
                kwargs["ncol"] = ncol

        # Coerce scatterpoints to int
        sp = kwargs.pop("scatterpoints", 1)
        try:
            sp = int(sp)
        except (ValueError, TypeError):
            sp = 1

        defaults = {
            'handletextpad': 0.3,
            'labelspacing': 0.5,
            'borderpad': 0.5,
            'borderaxespad': 0.5,
            'columnspacing': 0.6,
            'handlelength': 1.5,
            'markerscale': 0.9,
            'scatterpoints': sp,
            'fontsize': fontsize,
            'loc': loc,
            'ncol': ncol,
        }

        defaults.update(kwargs)
        legend = ax.legend(**defaults)

        if frame:
            legend.get_frame().set_linewidth(boxwidth)
            legend.get_frame().set_edgecolor(edgecolor)
        else:
            legend.get_frame().set_visible(False)

        return legend

    # ============== SUBPLOT / MULTI-PANEL SUPPORT ==============

    @staticmethod
    def _style_ax(ax):
        """Apply spine and tick styling to a single axes object."""
        for spine in ax.spines.values():
            spine.set_linewidth(PlotSettings.SPINE_WIDTH)
        ax.tick_params(
            axis='both', which='major', direction='in',
            width=PlotSettings.TICK_WIDTH, length=PlotSettings.TICK_LENGTH,
            labelsize=PlotSettings.TICK_LABELSIZE,
            bottom=True, top=True, left=True, right=True
        )
        ax.tick_params(
            axis='both', which='minor', direction='in',
            width=PlotSettings.MINOR_TICK_WIDTH, length=PlotSettings.MINOR_TICK_LENGTH,
            bottom=True, top=True, left=True, right=True
        )

    @staticmethod
    def plot_init_subplots(nrows=1, ncols=1, figsize=None, style='ieee',
                           sharex=False, sharey=False, **kwargs):
        """
        Create a figure with multiple subplots, all styled consistently.

        Parameters:
        -----------
        nrows, ncols : int
            Grid dimensions
        figsize : tuple, optional
            Figure size. If None, scales from PLOT_SIZE based on grid.
        style : str, optional
            Matplotlib style context (default: 'ieee')
        sharex, sharey : bool, optional
            Whether subplots share x/y axes
        **kwargs : dict
            Additional arguments passed to plt.subplots()

        Returns:
        --------
        tuple : (fig, axes)
        """
        if figsize is None:
            w, h = PlotSettings.PLOT_SIZE
            figsize = (w * ncols, h * nrows)

        PlotSettings._initialize_latex()

        with plt.style.context([style]):
            plt.rcParams['font.family'] = PlotSettings.GRAPHIC_FONT
            plt.rcParams['mathtext.fontset'] = PlotSettings.MATH_FONT
            plt.rcParams['text.usetex'] = True

            fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                     sharex=sharex, sharey=sharey, **kwargs)

            # Style all axes
            if nrows == 1 and ncols == 1:
                PlotSettings._style_ax(axes)
            else:
                for ax in np.atleast_1d(axes).flat:
                    PlotSettings._style_ax(ax)

            fig.subplots_adjust(wspace=PlotSettings.WSPACE,
                                hspace=PlotSettings.HSPACE)

            return fig, axes

    @staticmethod
    def set_shared_label(fig, xlabel=None, ylabel=None, fontsize=None):
        """
        Add a single shared x/y label for a grid of subplots.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
        xlabel, ylabel : str, optional
        fontsize : int, optional
        """
        if fontsize is None:
            fontsize = PlotSettings.LABEL_FONTSIZE
        if xlabel:
            fig.supxlabel(xlabel, fontsize=fontsize)
        if ylabel:
            fig.supylabel(ylabel, fontsize=fontsize)

    @staticmethod
    def adjust_subplots(fig, wspace=None, hspace=None, **kwargs):
        """
        Adjust subplot spacing with sensible defaults.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
        wspace, hspace : float, optional
        **kwargs : passed to fig.subplots_adjust()
        """
        if wspace is None:
            wspace = PlotSettings.WSPACE
        if hspace is None:
            hspace = PlotSettings.HSPACE
        fig.subplots_adjust(wspace=wspace, hspace=hspace, **kwargs)

    # ============== AXIS FORMATTING UTILITIES ==============

    @staticmethod
    def set_axis_labels(ax, xlabel=None, ylabel=None, fontsize=None):
        """
        Set x and y labels with the default LABEL_FONTSIZE.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
        xlabel, ylabel : str, optional
        fontsize : int, optional
        """
        if fontsize is None:
            fontsize = PlotSettings.LABEL_FONTSIZE
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=fontsize)

    @staticmethod
    def set_scientific_notation(ax, axis='y', scilimits=(0, 0)):
        """
        Enable scientific notation on an axis and style the offset text.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
        axis : str
            'x', 'y', or 'both'
        scilimits : tuple
            Range outside which scientific notation is used
        """
        axes_to_set = []
        if axis in ('x', 'both'):
            axes_to_set.append(ax.xaxis)
        if axis in ('y', 'both'):
            axes_to_set.append(ax.yaxis)

        for axis_obj in axes_to_set:
            formatter = mticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits(scilimits)
            axis_obj.set_major_formatter(formatter)
            axis_obj.offsetText.set_fontsize(PlotSettings.TICK_LABELSIZE)

    @staticmethod
    def set_log_scale(ax, axis='both', minor_ticks=True):
        """
        Set logarithmic scale with properly styled minor ticks.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
        axis : str
            'x', 'y', or 'both'
        minor_ticks : bool
            Whether to show minor ticks on log axes
        """
        if axis in ('x', 'both'):
            ax.set_xscale('log')
        if axis in ('y', 'both'):
            ax.set_yscale('log')

        if minor_ticks:
            ax.tick_params(
                axis='both', which='minor', direction='in',
                width=PlotSettings.MINOR_TICK_WIDTH,
                length=PlotSettings.MINOR_TICK_LENGTH,
                bottom=True, top=True, left=True, right=True
            )

    # ============== COLORBAR HELPER ==============

    @staticmethod
    def add_colorbar(fig, mappable, ax, label=None, orientation='vertical',
                     fontsize=None, tick_labelsize=None, **kwargs):
        """
        Add a consistently styled colorbar.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
        mappable : ScalarMappable (e.g. return from contourf, imshow)
        ax : matplotlib.axes.Axes
        label : str, optional
        orientation : str, optional ('vertical' or 'horizontal')
        fontsize : int, optional (label font size)
        tick_labelsize : int, optional
        **kwargs : passed to fig.colorbar()

        Returns:
        --------
        matplotlib.colorbar.Colorbar
        """
        if fontsize is None:
            fontsize = PlotSettings.LABEL_FONTSIZE
        if tick_labelsize is None:
            tick_labelsize = PlotSettings.TICK_LABELSIZE

        cbar = fig.colorbar(mappable, ax=ax, orientation=orientation, **kwargs)
        cbar.ax.tick_params(labelsize=tick_labelsize,
                            width=PlotSettings.TICK_WIDTH,
                            length=PlotSettings.TICK_LENGTH,
                            direction='in')
        cbar.outline.set_linewidth(PlotSettings.SPINE_WIDTH)
        if label:
            cbar.set_label(label, fontsize=fontsize)
        return cbar

    # ============== COLOR UTILITIES ==============

    @staticmethod
    def get_cmap(name='viridis', n=5):
        """
        Return n evenly-spaced colors from a matplotlib colormap.

        Parameters:
        -----------
        name : str
            Colormap name (default: 'viridis')
        n : int
            Number of colors to return

        Returns:
        --------
        list : List of RGBA tuples
        """
        cmap = mpl.colormaps[name]
        return [cmap(i / max(n - 1, 1)) for i in range(n)]

    @staticmethod
    def lighten_color(color, amount=0.3):
        """
        Lighten a color by blending towards white.

        Parameters:
        -----------
        color : str or tuple
            Any matplotlib-compatible color
        amount : float
            Blend fraction towards white (0 = unchanged, 1 = white)

        Returns:
        --------
        tuple : RGB tuple
        """
        c = np.array(mcolors.to_rgb(color))
        white = np.array([1.0, 1.0, 1.0])
        return tuple(c + (white - c) * amount)

    @staticmethod
    def darken_color(color, amount=0.3):
        """
        Darken a color by blending towards black.

        Parameters:
        -----------
        color : str or tuple
            Any matplotlib-compatible color
        amount : float
            Blend fraction towards black (0 = unchanged, 1 = black)

        Returns:
        --------
        tuple : RGB tuple
        """
        c = np.array(mcolors.to_rgb(color))
        return tuple(c * (1.0 - amount))

    @staticmethod
    def set_color_cycle(ax, palette=None):
        """
        Set a color cycle on an axes from a palette list.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
        palette : list, optional
            List of color specs. If None, uses PlotSettings.COLORS
        """
        if palette is None:
            palette = PlotSettings.COLORS
        from cycler import cycler
        ax.set_prop_cycle(cycler(color=palette))

    # ============== ANNOTATION / INSET HELPERS ==============

    @staticmethod
    def add_subplot_label(ax, label='(a)', loc='upper left', fontsize=None,
                          fontweight=None, offset=(0.03, 0.95)):
        """
        Place a panel label like (a), (b), (c) on an axes.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
        label : str
        loc : str (unused, kept for API compatibility; use offset to position)
        fontsize : int, optional
        fontweight : str, optional
        offset : tuple
            (x, y) in axes fraction coordinates
        """
        if fontsize is None:
            fontsize = PlotSettings.SUBPLOT_LABEL_FONTSIZE
        if fontweight is None:
            fontweight = PlotSettings.SUBPLOT_LABEL_FONTWEIGHT
        ax.text(offset[0], offset[1], label,
                transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight,
                va='top', ha='left')

    @staticmethod
    def add_inset(ax, bounds, style_inset=True):
        """
        Create a styled inset axes.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Parent axes
        bounds : tuple
            (x, y, width, height) in axes fraction coordinates
        style_inset : bool
            Whether to apply PlotSettings styling to the inset

        Returns:
        --------
        matplotlib.axes.Axes
            The inset axes
        """
        inset_ax = ax.inset_axes(bounds)
        if style_inset:
            PlotSettings._style_ax(inset_ax)
            inset_ax.tick_params(labelsize=PlotSettings.TICK_LABELSIZE - 2)
        return inset_ax

    # ============== ERROR BAND / ERRORBAR PLOTTING ==============

    @staticmethod
    def plot_with_fill(ax, x, y, yerr, color=None, alpha=None,
                       label=None, linewidth=None, **kwargs):
        """
        Plot a line with a shaded error band.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
        x, y : array-like
        yerr : array-like or tuple of (lower, upper)
            If 1-D, symmetric band; if tuple, asymmetric.
        color : str, optional
        alpha : float, optional (fill transparency)
        label : str, optional
        linewidth : float, optional
        **kwargs : passed to ax.plot()

        Returns:
        --------
        tuple : (line, fill)
        """
        if color is None:
            color = PlotSettings.COLORS[0]
        if alpha is None:
            alpha = PlotSettings.ALPHA
        if linewidth is None:
            linewidth = PlotSettings.LINEWIDTH

        x, y = np.asarray(x), np.asarray(y)

        if isinstance(yerr, tuple) and len(yerr) == 2:
            lower, upper = np.asarray(yerr[0]), np.asarray(yerr[1])
        else:
            yerr = np.asarray(yerr)
            lower, upper = y - yerr, y + yerr

        line, = ax.plot(x, y, color=color, linewidth=linewidth, label=label, **kwargs)
        fill = ax.fill_between(x, lower, upper, color=color, alpha=alpha)
        return line, fill

    @staticmethod
    def plot_with_errorbars(ax, x, y, yerr=None, xerr=None, color=None,
                            marker=None, label=None, linewidth=None,
                            markersize=None, capsize=None, markeredgewidth=None,
                            linestyle='none', **kwargs):
        """
        Plot data with error bars using PlotSettings defaults.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
        x, y : array-like
        yerr, xerr : array-like, optional
        color : str, optional
        marker : str, optional
        label : str, optional
        linewidth : float, optional
        markersize : float, optional
        capsize : float, optional
        markeredgewidth : float, optional
        linestyle : str, optional
        **kwargs : passed to ax.errorbar()

        Returns:
        --------
        ErrorbarContainer
        """
        if color is None:
            color = PlotSettings.COLORS[0]
        if marker is None:
            marker = PlotSettings.MARKERS[0]
        if linewidth is None:
            linewidth = PlotSettings.LINEWIDTH
        if markersize is None:
            markersize = PlotSettings.MARKERSIZE
        if capsize is None:
            capsize = PlotSettings.CAPSIZE
        if markeredgewidth is None:
            markeredgewidth = PlotSettings.MARKEREDGEWIDTH

        return ax.errorbar(
            x, y, yerr=yerr, xerr=xerr,
            fmt=marker, color=color, markersize=markersize,
            capsize=capsize, markeredgewidth=markeredgewidth,
            elinewidth=linewidth, label=label, linestyle=linestyle,
            **kwargs
        )

    # ============== TWIN AXIS SUPPORT ==============

    @staticmethod
    def add_twin_axis(ax, side='right', color=None, ylabel=None, fontsize=None):
        """
        Create a styled twin axis (twinx or twiny).

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
        side : str
            'right' for twinx, 'top' for twiny
        color : str, optional
            If given, colors the spine and tick labels on the twin side
        ylabel : str, optional
            Label for the twin axis
        fontsize : int, optional

        Returns:
        --------
        matplotlib.axes.Axes
            The twin axes
        """
        if fontsize is None:
            fontsize = PlotSettings.LABEL_FONTSIZE

        if side == 'right':
            twin = ax.twinx()
        else:
            twin = ax.twiny()

        PlotSettings._style_ax(twin)

        if color:
            spine_key = side
            twin.spines[spine_key].set_color(color)
            twin.tick_params(axis='y' if side == 'right' else 'x',
                             colors=color)
            if ylabel:
                twin.set_ylabel(ylabel, fontsize=fontsize, color=color)
        elif ylabel:
            twin.set_ylabel(ylabel, fontsize=fontsize)

        return twin

    # ============== TEXTBOX HELPER ==============

    @staticmethod
    def add_textbox(ax, text, loc='upper left', fontsize=None, alpha=None,
                    boxstyle='round,pad=0.4', edgecolor='black',
                    facecolor='white'):
        """
        Place a styled annotation text box on the axes.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
        text : str
        loc : str
            Predefined positions: 'upper left', 'upper right',
            'lower left', 'lower right', 'center'
        fontsize : int, optional
        alpha : float, optional (box transparency)
        boxstyle : str, optional
        edgecolor, facecolor : str, optional

        Returns:
        --------
        matplotlib.text.Annotation
        """
        if fontsize is None:
            fontsize = PlotSettings.TEXTBOX_FONTSIZE
        if alpha is None:
            alpha = PlotSettings.TEXTBOX_ALPHA

        loc_map = {
            'upper left':  (0.05, 0.95, 'top', 'left'),
            'upper right': (0.95, 0.95, 'top', 'right'),
            'lower left':  (0.05, 0.05, 'bottom', 'left'),
            'lower right': (0.95, 0.05, 'bottom', 'right'),
            'center':      (0.50, 0.50, 'center', 'center'),
        }
        x, y, va, ha = loc_map.get(loc, (0.05, 0.95, 'top', 'left'))

        bbox_props = dict(boxstyle=boxstyle, facecolor=facecolor,
                          edgecolor=edgecolor, alpha=alpha)
        return ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
                       verticalalignment=va, horizontalalignment=ha,
                       bbox=bbox_props)

    @staticmethod
    def save_figure(fig, filename, dpi=None):
        """
        Save a figure with the predefined resolution.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            The figure object to save
        filename : str
            Output filename (can include path)
        dpi : int, optional
            Resolution in dots per inch.
            If None, uses PlotSettings.RESOLUTION
        """
        if dpi is None:
            dpi = PlotSettings.RESOLUTION

        output_dir = os.getcwd()
        file_path = os.path.join(output_dir, filename)

        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')

    @staticmethod
    def save_plot(fig, filename_base, folder="PLOTS", dpi=None,
                  formats=('png', 'pdf')):
        """
        Save a figure in multiple formats.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            The figure object to save
        filename_base : str
            Base filename without extension
        folder : str, optional
            Output folder name (default: 'PLOTS')
        dpi : int, optional
            Resolution in dots per inch.
            If None, uses PlotSettings.RESOLUTION
        formats : tuple of str, optional
            File formats to save (default: ('png', 'pdf')).
            Supported: 'png', 'pdf', 'svg', 'eps'
        """
        os.makedirs(folder, exist_ok=True)

        for fmt in formats:
            path = os.path.join(folder, f"{filename_base}.{fmt}")
            PlotSettings.save_figure(fig, path, dpi=dpi)

    @staticmethod
    def save_all_open_figures(folder="PLOTS", filename_prefix="fig",
                              dpi=None, formats=('png', 'pdf')):
        """
        Batch-save all currently open matplotlib figures.

        Parameters:
        -----------
        folder : str, optional
        filename_prefix : str, optional
        dpi : int, optional
        formats : tuple of str, optional
        """
        os.makedirs(folder, exist_ok=True)
        for i, fig_num in enumerate(plt.get_fignums()):
            fig = plt.figure(fig_num)
            base = f"{filename_prefix}_{i + 1}"
            PlotSettings.save_plot(fig, base, folder=folder, dpi=dpi,
                                   formats=formats)

    @staticmethod
    def set_all_rcparams():
        """Apply all plot settings to matplotlib rcParams globally."""
        PlotSettings._initialize_latex()
        plt.rcParams['font.family'] = PlotSettings.GRAPHIC_FONT
        plt.rcParams['mathtext.fontset'] = PlotSettings.MATH_FONT
        plt.rcParams['lines.linewidth'] = PlotSettings.LINEWIDTH
        plt.rcParams['lines.markersize'] = PlotSettings.MARKERSIZE

        # Font embedding for editable text in PDF/EPS
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        # Spine and axes
        plt.rcParams['axes.linewidth'] = PlotSettings.SPINE_WIDTH
        plt.rcParams['axes.labelsize'] = PlotSettings.LABEL_FONTSIZE

        # Ticks
        plt.rcParams['xtick.labelsize'] = PlotSettings.TICK_LABELSIZE
        plt.rcParams['ytick.labelsize'] = PlotSettings.TICK_LABELSIZE
        plt.rcParams['xtick.major.width'] = PlotSettings.TICK_WIDTH
        plt.rcParams['ytick.major.width'] = PlotSettings.TICK_WIDTH
        plt.rcParams['xtick.major.size'] = PlotSettings.TICK_LENGTH
        plt.rcParams['ytick.major.size'] = PlotSettings.TICK_LENGTH
        plt.rcParams['xtick.minor.width'] = PlotSettings.MINOR_TICK_WIDTH
        plt.rcParams['ytick.minor.width'] = PlotSettings.MINOR_TICK_WIDTH
        plt.rcParams['xtick.minor.size'] = PlotSettings.MINOR_TICK_LENGTH
        plt.rcParams['ytick.minor.size'] = PlotSettings.MINOR_TICK_LENGTH
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        # Legend
        plt.rcParams['legend.fontsize'] = PlotSettings.LEGEND_FONTSIZE

        # DPI
        plt.rcParams['figure.dpi'] = PlotSettings.SCREEN_DPI
        plt.rcParams['savefig.dpi'] = PlotSettings.RESOLUTION


# ============== BACKWARD COMPATIBILITY WRAPPERS ==============
# Module-level functions for backward compatibility with original PLOT_SETTINGS.py
# These allow using the old import style: from plot_settings import plot_init, style_legend, etc.

def plot_init(figsize=None, style='ieee'):
    """Backward compatible wrapper for PlotSettings.plot_init()"""
    return PlotSettings.plot_init(figsize=figsize, style=style)


def style_legend(ax, loc="upper right", ncol=1, edgecolor="black",
                 frame=True, fontsize=None, boxwidth=None, **kwargs):
    """Backward compatible wrapper for PlotSettings.style_legend()"""
    return PlotSettings.style_legend(
        ax, loc=loc, ncol=ncol, edgecolor=edgecolor,
        frame=frame, fontsize=fontsize, boxwidth=boxwidth, **kwargs
    )


def save_figure(fig, filename, dpi=None):
    """Backward compatible wrapper for PlotSettings.save_figure()"""
    return PlotSettings.save_figure(fig, filename, dpi=dpi)


def save_plot(fig, filename_base, folder="PLOTS", dpi=None, formats=('png', 'pdf')):
    """Backward compatible wrapper for PlotSettings.save_plot()"""
    return PlotSettings.save_plot(fig, filename_base, folder=folder, dpi=dpi,
                                  formats=formats)


# Module-level constants for backward compatibility
plot_size = PlotSettings.PLOT_SIZE
colors = PlotSettings.COLORS
color_10 = PlotSettings.COLOR_10
color_8 = PlotSettings.COLOR_8
dual_colors = PlotSettings.DUAL_COLORS
markers = PlotSettings.MARKERS

graphic_font = PlotSettings.GRAPHIC_FONT
math_font = PlotSettings.MATH_FONT
spine_width = PlotSettings.SPINE_WIDTH
markersize = PlotSettings.MARKERSIZE
capsize = PlotSettings.CAPSIZE
markeredgewidth = PlotSettings.MARKEREDGEWIDTH
legend_linewidth = PlotSettings.LEGEND_LINEWIDTH
linewidth = PlotSettings.LINEWIDTH
tick_width = PlotSettings.TICK_WIDTH
tick_length = PlotSettings.TICK_LENGTH
minor_tick_width = PlotSettings.MINOR_TICK_WIDTH
minor_tick_length = PlotSettings.MINOR_TICK_LENGTH
tick_labelsize = PlotSettings.TICK_LABELSIZE
legend_fontsize = PlotSettings.LEGEND_FONTSIZE
legend_boxwidth = PlotSettings.LEGEND_BOXWIDTH
label_fontsize = PlotSettings.LABEL_FONTSIZE
borderaxespad = PlotSettings.BORDERAXESPAD
alpha = PlotSettings.ALPHA
resolution_value = PlotSettings.RESOLUTION

# Backward compatible face_colors function
def face_colors(colors=None, alpha_val=None):
    """Backward compatible wrapper for PlotSettings.get_face_colors()"""
    if colors is None:
        colors = PlotSettings.COLORS
    if alpha_val is None:
        alpha_val = PlotSettings.ALPHA
    return PlotSettings.get_face_colors(colors=colors, alpha=alpha_val)


# New backward compatible wrappers
def plot_init_subplots(nrows=1, ncols=1, figsize=None, style='ieee', **kwargs):
    """Backward compatible wrapper for PlotSettings.plot_init_subplots()"""
    return PlotSettings.plot_init_subplots(nrows=nrows, ncols=ncols,
                                           figsize=figsize, style=style, **kwargs)


def set_shared_label(fig, xlabel=None, ylabel=None, fontsize=None):
    """Backward compatible wrapper for PlotSettings.set_shared_label()"""
    return PlotSettings.set_shared_label(fig, xlabel=xlabel, ylabel=ylabel,
                                         fontsize=fontsize)


def adjust_subplots(fig, wspace=None, hspace=None, **kwargs):
    """Backward compatible wrapper for PlotSettings.adjust_subplots()"""
    return PlotSettings.adjust_subplots(fig, wspace=wspace, hspace=hspace, **kwargs)


def set_axis_labels(ax, xlabel=None, ylabel=None, fontsize=None):
    """Backward compatible wrapper for PlotSettings.set_axis_labels()"""
    return PlotSettings.set_axis_labels(ax, xlabel=xlabel, ylabel=ylabel,
                                        fontsize=fontsize)


def set_scientific_notation(ax, axis='y', scilimits=(0, 0)):
    """Backward compatible wrapper for PlotSettings.set_scientific_notation()"""
    return PlotSettings.set_scientific_notation(ax, axis=axis, scilimits=scilimits)


def set_log_scale(ax, axis='both', minor_ticks=True):
    """Backward compatible wrapper for PlotSettings.set_log_scale()"""
    return PlotSettings.set_log_scale(ax, axis=axis, minor_ticks=minor_ticks)


def add_colorbar(fig, mappable, ax, label=None, **kwargs):
    """Backward compatible wrapper for PlotSettings.add_colorbar()"""
    return PlotSettings.add_colorbar(fig, mappable, ax, label=label, **kwargs)


def get_cmap(name='viridis', n=5):
    """Backward compatible wrapper for PlotSettings.get_cmap()"""
    return PlotSettings.get_cmap(name=name, n=n)


def lighten_color(color, amount=0.3):
    """Backward compatible wrapper for PlotSettings.lighten_color()"""
    return PlotSettings.lighten_color(color, amount=amount)


def darken_color(color, amount=0.3):
    """Backward compatible wrapper for PlotSettings.darken_color()"""
    return PlotSettings.darken_color(color, amount=amount)


def set_color_cycle(ax, palette=None):
    """Backward compatible wrapper for PlotSettings.set_color_cycle()"""
    return PlotSettings.set_color_cycle(ax, palette=palette)


def add_subplot_label(ax, label='(a)', **kwargs):
    """Backward compatible wrapper for PlotSettings.add_subplot_label()"""
    return PlotSettings.add_subplot_label(ax, label=label, **kwargs)


def add_inset(ax, bounds, style_inset=True):
    """Backward compatible wrapper for PlotSettings.add_inset()"""
    return PlotSettings.add_inset(ax, bounds, style_inset=style_inset)


def plot_with_fill(ax, x, y, yerr, **kwargs):
    """Backward compatible wrapper for PlotSettings.plot_with_fill()"""
    return PlotSettings.plot_with_fill(ax, x, y, yerr, **kwargs)


def plot_with_errorbars(ax, x, y, **kwargs):
    """Backward compatible wrapper for PlotSettings.plot_with_errorbars()"""
    return PlotSettings.plot_with_errorbars(ax, x, y, **kwargs)


def add_twin_axis(ax, side='right', color=None, ylabel=None, fontsize=None):
    """Backward compatible wrapper for PlotSettings.add_twin_axis()"""
    return PlotSettings.add_twin_axis(ax, side=side, color=color,
                                      ylabel=ylabel, fontsize=fontsize)


def add_textbox(ax, text, loc='upper left', **kwargs):
    """Backward compatible wrapper for PlotSettings.add_textbox()"""
    return PlotSettings.add_textbox(ax, text, loc=loc, **kwargs)


def save_all_open_figures(folder="PLOTS", **kwargs):
    """Backward compatible wrapper for PlotSettings.save_all_open_figures()"""
    return PlotSettings.save_all_open_figures(folder=folder, **kwargs)


# ============== EXAMPLE USAGE ==============
if __name__ == "__main__":
    # Create a figure with predefined settings
    fig, ax = PlotSettings.plot_init()

    # Plot some example data
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 2, 3, 5]
    colors = PlotSettings.COLOR_10[:5]

    ax.plot(x, y, marker='o', color=colors[0], linewidth=PlotSettings.LINEWIDTH)
    ax.set_xlabel('X Label', fontsize=PlotSettings.LABEL_FONTSIZE)
    ax.set_ylabel('Y Label', fontsize=PlotSettings.LABEL_FONTSIZE)

    # Style the legend
    ax.plot([], [], label='Example Data')
    PlotSettings.style_legend(ax, loc='upper left')

    # Save the figure
    PlotSettings.save_plot(fig, 'example_plot')

    print("Example plot saved to PLOTS/")
