#!/usr/bin/env python
"""Standalone *shared* colourbars for the Figure-11 2x2 (GPR/SVGP x mean/STD).

Emits two slim bars (PNG + PDF) to SHARED_COLORBAR/:
  * MEAN : <Delta gamma>      Blues_r, [-5, 0]   -> shared_residual_colorbar_*
  * STD  : sigma_{Delta gamma} Blues_r, [0, SIGMA_VMAX] -> shared_std_colorbar_*

The MEAN bar sits above the left (mean) column, the STD bar above the right
(sigma) column.  Both 2D surfaces use Blues_r so the two bars match in hue.

IMPORTANT: SIGMA_VMAX here MUST equal SIGMA_VMAX in the two preview notebooks
(GPR/preview_diff_std.ipynb, SVGP/preview_diff_std.ipynb) so the right-column
panels and the STD bar share one scale.

Run:  PATH=<texlive>:$PATH /home/darshan/A6/py_A6/bin/python make_shared_colorbar.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import thermoift.PLOT_SETTINGS as ps

OUTDIR     = "SHARED_COLORBAR"
LABEL_FS   = ps.label_fontsize * 1.3   # tune if text looks small after LaTeX scaling
TICK_FS    = ps.label_fontsize * 1.1
THICK      = 0.2    # band THICKNESS (slim = smaller); fraction of the figure's short side
BARLEN     = 0.92   # band length as a fraction of the figure's long side
OUTLINE_LW = 0.8    # colorbar border line width

# shared sigma (STD) range — keep in sync with the preview notebooks
SIGMA_VMIN, SIGMA_VMAX = 0.0, 2.5


def _mappable(cmap, vmin, vmax):
    cm = plt.get_cmap(cmap).copy()
    cm.set_under(cm(0.0))   # low-end arrow colour
    cm.set_over(cm(1.0))    # high-end arrow colour
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cm)
    sm.set_array([])
    return sm


def make(orientation, figsize, fname, *, cmap, vmin, vmax, label, ticks=None, extend="both"):
    with plt.style.context(["ieee"]):
        plt.rcParams["font.family"]      = ps.graphic_font
        plt.rcParams["mathtext.fontset"] = ps.math_font
        plt.rcParams["text.usetex"]      = True

        fig = plt.figure(figsize=figsize)
        if orientation == "horizontal":
            cax = fig.add_axes([(1 - BARLEN) / 2, 0.34, BARLEN, THICK])
        else:
            cax = fig.add_axes([0.34, (1 - BARLEN) / 2, THICK, BARLEN])
        cbar = fig.colorbar(_mappable(cmap, vmin, vmax), cax=cax,
                            orientation=orientation, extend=extend)
        if ticks is not None:
            cbar.set_ticks(ticks)
        ps.style_colorbar(cbar)
        cbar.ax.tick_params(labelsize=TICK_FS)
        cbar.outline.set_linewidth(OUTLINE_LW)
        if orientation == "vertical":
            cbar.set_label(label, fontsize=LABEL_FS, rotation=90, labelpad=10)
        else:
            cbar.ax.xaxis.set_label_position("top")   # caption on top of the bar
            cbar.set_label(label, fontsize=LABEL_FS, labelpad=8)
        ps.save_plot(fig, fname, folder=OUTDIR)
        plt.close(fig)


MEAN = dict(cmap="Blues_r", vmin=-5.0, vmax=0.0, ticks=[-4, -3, -2, -1], extend="both",
            label=r"$\langle\Delta\gamma\rangle\ /\ [\mathrm{mN}\,\mathrm{m}^{-1}]$")
STD  = dict(cmap="Blues", vmin=SIGMA_VMIN, vmax=SIGMA_VMAX, ticks=None, extend="max",
            label=r"$\sigma_{\Delta\gamma}\ /\ [\mathrm{mN}\,\mathrm{m}^{-1}]$")


if __name__ == "__main__":
    make("horizontal", (5.5, 1.0), "shared_residual_colorbar_horizontal", **MEAN)
    make("vertical",   (1.0, 4.5), "shared_residual_colorbar_vertical",   **MEAN)
    make("horizontal", (5.5, 1.0), "shared_std_colorbar_horizontal",      **STD)
    make("vertical",   (1.0, 4.5), "shared_std_colorbar_vertical",        **STD)
    print("Wrote shared colourbars ->", os.path.abspath(OUTDIR))
