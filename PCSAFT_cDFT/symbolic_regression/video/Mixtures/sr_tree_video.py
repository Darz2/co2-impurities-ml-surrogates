#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "pillow",
#     "imageio-ffmpeg",
# ]
# ///
# ============================================================
#  Symbolic-regression "discovery" animation for the MIXTURE runs.
#
#  Reproduces the manim-style demo (expression tree on the left, a
#  Truth-vs-Prediction plot on the right) but for the multi-feature
#  mixture model. We walk the Pareto front in hall_of_fame_equations.csv
#  (complexity up, loss down); for each equation we
#     - parse equation_raw into an expression tree and draw it, and
#     - evaluate it on the data to get the predicted target, then map
#       eps_base -> gamma via  gamma = gamma_base * (1 + eps_base),
#       and show Truth (cyan) vs Prediction (orange) in a 3D plot whose
#       floor axes are Tr (x) and Pr (y); the other features are left
#       off the axes (as requested) but still drive the target.
#
#  The safe_* operators and the eps_base->gamma map were verified to
#  reproduce the saved Predicted / gamma_cDFT_pred columns exactly.
#
#  Frames are rendered to PNGs in parallel across all CPU cores (one forked
#  worker per core, --jobs to override) and encoded once with ffmpeg, so a
#  full re-render is a few seconds rather than a minute.
#
#  Usage (uv handles deps from the inline metadata above; --no-project
#  skips the A6 workspace so the ephemeral env resolves cleanly):
#     uv run --no-project sr_tree_video.py --outdir SR_MIXTURES_OUTPUTS    --label V1
#     uv run --no-project sr_tree_video.py --outdir SR_MIXTURES_OUTPUTS_V2 --label V2 \
#         --tp-from SR_MIXTURES_OUTPUTS
#  Useful knobs: --jobs N (worker count), --dpi (frame resolution),
#  --format gif|mp4|both, --fps, --morph, --hold, --nsample.
# ============================================================
import argparse
import ast
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Render math the way LaTeX does (we are NOT using a real TeX install -- this is
# matplotlib's built-in mathtext). "cm" = Computer Modern, so $T$, $P$, $\gamma$
# come out in the familiar serif italic; a serif face keeps tick numbers matching.
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"

# Use the bundled ffmpeg from imageio-ffmpeg so MP4/GIF export needs no system install.
try:
    import imageio_ffmpeg
    _HAVE_FFMPEG = True
except Exception:
    _HAVE_FFMPEG = False

HERE = Path(__file__).resolve().parent

# ---------- colours / theme (light) ----------
BG = "#ffffff"        # background: white
FG = "#000000"        # text / axes / edges: black
TRUTH_C = "#ef6c00"   # orange (darkened so it reads on white)
PRED_C = "#1597a5"    # cyan (darkened so it reads on white)
NODE_FC = "#eef2f7"   # light node fill so black text/edges show
NODE_EC = "#000000"   # node outline: black
ACCENT = "#c77f00"    # (unused) old gold highlight; selected model now uses PRED_C

# CO2 critical constants used by Mixture.jl to build Tr = T/Tc, Pr = P/Pc.
# (constant columns in ../../CombinedDatasetSEC_A4.csv); used to put raw
# T [K] and P [bar] on the plot axes instead of the reduced Tr, Pr.
TC_CO2 = 302.71   # K
PC_CO2 = 75.512   # bar

# ---------------------------------------------------------------
# safe operators (match SymbolicRegression.jl; verified vs saved preds)
# ---------------------------------------------------------------
def safe_log(x):
    return np.log(np.abs(x))


def safe_sqrt(x):
    return np.sqrt(np.abs(x))


def safe_div(a, b):
    b = np.where(b == 0.0, np.nan, b)
    return a / b


def safepow(a, b):
    return np.sign(a) * np.abs(a) ** b


SAFE_NS = {
    "safe_log": safe_log,
    "safe_sqrt": safe_sqrt,
    "safe_div": safe_div,
    "safepow": safepow,
    "abs": np.abs,
}

# ---------------------------------------------------------------
# pretty symbols for variables (mathtext, used in nodes and title)
# ---------------------------------------------------------------
# Values are BARE mathtext fragments (no $...$): build() wraps node labels in
# $...$ and the title embeds them inside one outer $...$, so adding $ here
# would double up ($$ = empty math) and render the LaTeX as literal text.
SYM = {
    "delta_rho_norm": r"\Delta\rho^{*}",
    "delta_rho": r"\Delta\rho",
    "x_total_impurity": r"x_i",
    "y_total_impurity": r"y_i",
    "z_total_impurity": r"z_i",
    "delta_total_impurity": r"\Delta_{\mathrm{imp}}",
    "logK_total_impurity": r"\ln\!K_i",
    "Tr": r"T_r",
    "oneMinusTr": r"(1\!-\!T_r)",
    "Pr": r"P_r",
    "P_over_Psat0": r"P/P_0",
    "gamma_base_ratio": r"\gamma_r",
}


def vsym(name):
    return SYM.get(name, name.replace("_", r"\_"))


# ---------------------------------------------------------------
# expression tree
# ---------------------------------------------------------------
class Node:
    __slots__ = ("label", "math", "children", "x", "y")

    def __init__(self, label, math, children=None):
        self.label = label          # mathtext string drawn inside the node
        self.math = math            # mathtext for the full-equation title
        self.children = children or []
        self.x = 0.0
        self.y = 0.0


def _fmt_const(v):
    if v == int(v):
        return str(int(v))
    return f"{v:.3g}"


BINOP = {ast.Add: ("+", "+"), ast.Sub: ("-", "-"),
         ast.Mult: (r"\times", r"\cdot "), ast.Div: (r"\div", "/")}


def build(node):
    """ast node -> Node (label/math for display)."""
    if isinstance(node, ast.Expression):
        return build(node.body)
    if isinstance(node, ast.BinOp):
        nlab, mop = BINOP[type(node.op)]
        l, r = build(node.left), build(node.right)
        math = f"({l.math} {mop} {r.math})" if mop != "/" else rf"\frac{{{l.math}}}{{{r.math}}}"
        return Node(f"${nlab}$", math, [l, r])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = "-" if isinstance(node.op, ast.USub) else "+"
        if isinstance(node.operand, ast.Constant):
            v = -node.operand.value if sign == "-" else node.operand.value
            return Node(f"${_fmt_const(v)}$", _fmt_const(v))
        c = build(node.operand)
        return Node(f"${sign}$", f"({sign}{c.math})", [c])
    if isinstance(node, ast.Call):
        fn = node.func.id
        args = [build(a) for a in node.args]
        if fn == "safe_div":
            return Node(r"$\div$", rf"\frac{{{args[0].math}}}{{{args[1].math}}}", args)
        if fn == "safepow":
            return Node(r"$\wedge$", f"{{{args[0].math}}}^{{{args[1].math}}}", args)
        if fn == "safe_log":
            return Node(r"$\log$", rf"\log({args[0].math})", args)
        if fn == "safe_sqrt":
            return Node(r"$\sqrt{\ }$", rf"\sqrt{{{args[0].math}}}", args)
        if fn == "abs":
            return Node(r"$|\,\cdot\,|$", rf"\left|{args[0].math}\right|", args)
        # fallback: unknown unary fn
        return Node(rf"$\mathrm{{{fn}}}$", rf"\mathrm{{{fn}}}({args[0].math})", args)
    if isinstance(node, ast.Name):
        s = vsym(node.id)
        return Node(f"${s}$", s)
    if isinstance(node, ast.Constant):
        s = _fmt_const(float(node.value))
        return Node(f"${s}$", s)
    raise ValueError(f"unhandled ast node: {ast.dump(node)}")


def parse_eq(eq_raw):
    return build(ast.parse(eq_raw, mode="eval"))


# ---- layout: assign leaf x in order, internal x = mean(children) ----
# Raw coords only (leaf index across, -depth down); the actual placement onto
# the canvas is done globally in build_scene so every tree shares one scale and
# stays centred (a single node sits dead centre, not in a corner).
def layout(root):
    leaf = [0]

    def depth_assign(n, d):
        n.y = float(-d)
        if not n.children:
            n.x = float(leaf[0])
            leaf[0] += 1
        else:
            for c in n.children:
                depth_assign(c, d + 1)
            n.x = sum(c.x for c in n.children) / len(n.children)

    depth_assign(root, 0)
    xs, ys = [], []

    def collect(n):
        xs.append(n.x)
        ys.append(n.y)
        for c in n.children:
            collect(c)

    collect(root)
    return root, (min(xs), max(xs), min(ys), max(ys))  # raw extent (x0,x1,y0,y1)


def eval_eq(eq_raw, feat):
    ns = dict(SAFE_NS)
    ns.update(feat)
    return eval(eq_raw, {"__builtins__": {}}, ns)


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="SR output dir with hall_of_fame + predictions")
    ap.add_argument("--label", required=True, help="version label, e.g. V1")
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--nsample", type=int, default=700, help="points shown in the 3D cloud")
    ap.add_argument("--max-complexity", type=int, default=12,
                    help="stop walking the Pareto front past this complexity")
    ap.add_argument("--tp-from", default="SR_MIXTURES_OUTPUTS",
                    help="dir to borrow Tr/Pr from when this run's CSV lacks them (same split)")
    ap.add_argument("--tc-co2", type=float, default=TC_CO2, help="Tc of CO2 [K] for T = Tr*Tc")
    ap.add_argument("--pc-co2", type=float, default=PC_CO2, help="Pc of CO2 [bar] for P = Pr*Pc")
    ap.add_argument("--fps", type=int, default=10, help="playback frame rate (lower = slower)")
    ap.add_argument("--morph", type=int, default=10, help="frames to morph between equations")
    ap.add_argument("--hold", type=int, default=8, help="frames to hold on each equation")
    ap.add_argument("--format", default="both", choices=["both", "gif", "mp4"],
                    help="output format(s)")
    ap.add_argument("--out", default=None, help="explicit output path (extension sets format)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel worker processes for frame rendering (0 = all CPUs)")
    ap.add_argument("--dpi", type=int, default=120, help="render resolution for the frames")
    ap.add_argument("--keep-frames", action="store_true",
                    help="also save the individual PNG frames into <stem>_frames/")
    args = ap.parse_args()
    run(args)


def build_scene(args):

    outdir = (HERE / args.outdir) if not Path(args.outdir).is_absolute() else Path(args.outdir)
    stem = Path(args.out).with_suffix("") if args.out else HERE / f"SR_tree_video_{args.label}"

    # ---- data ----
    fname = f"SR_{args.split}_predictions_eps_base_with_features.csv"
    df = pd.read_csv(outdir / fname)
    rng = np.random.default_rng(7)
    sidx = np.sort(rng.choice(len(df), size=min(args.nsample, len(df)), replace=False))
    sub = df.iloc[sidx].reset_index(drop=True)
    feat = {c: sub[c].to_numpy(float) for c in sub.columns}
    gamma_base = sub["gamma_base"].to_numpy(float)
    gamma_truth = sub["gamma_cDFT_actual"].to_numpy(float)

    # raw T, P for the axes. Some feature sets (e.g. V2) don't carry Tr/Pr;
    # borrow them from --tp-from, which uses the SAME (seeded) split so rows align.
    if "Tr" in df.columns:
        tp = sub
    else:
        tp_dir = (HERE / args.tp_from) if not Path(args.tp_from).is_absolute() else Path(args.tp_from)
        tp_full = pd.read_csv(tp_dir / fname)
        if len(tp_full) != len(df) or not np.allclose(
                tp_full["gamma_base"].to_numpy(float), df["gamma_base"].to_numpy(float)):
            raise SystemExit(f"--tp-from rows don't align with {args.label}; cannot map T,P")
        tp = tp_full.iloc[sidx].reset_index(drop=True)
    # reduced -> raw, using the CO2 reference constants from Mixture.jl
    T_K = tp["Tr"].to_numpy(float) * args.tc_co2     # Kelvin
    P_bar = tp["Pr"].to_numpy(float) * args.pc_co2   # bar

    metrics = json.loads((outdir / "SR_residual_metrics.json").read_text())
    sel_complexity = int(metrics["complexity"])

    # ---- hall of fame (Pareto front) ----
    hof = pd.read_csv(outdir / "hall_of_fame_equations.csv")
    hof = hof[hof["complexity"] <= args.max_complexity].reset_index(drop=True)

    stages = []  # list of dicts: tree, gamma_pred, complexity, loss, is_selected, title
    for _, row in hof.iterrows():
        eq_raw = row["equation_raw"]
        eps_pred = np.asarray(eval_eq(eq_raw, feat), dtype=float)
        eps_pred = np.broadcast_to(eps_pred, gamma_base.shape).astype(float)
        gamma_pred = gamma_base * (1.0 + eps_pred)
        root, extent = layout(parse_eq(eq_raw))
        # R2 vs truth (on shown points)
        ss_res = np.nansum((gamma_truth - gamma_pred) ** 2)
        ss_tot = np.nansum((gamma_truth - np.mean(gamma_truth)) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        stages.append(dict(
            tree=root,
            extent=extent,
            gamma_pred=gamma_pred,
            complexity=int(row["complexity"]),
            loss=float(row["loss"]),
            r2=r2,
            title=rf"$\epsilon = {root.math}$",
            selected=(int(row["complexity"]) == sel_complexity),
        ))

    # ---- place every tree on ONE shared, centred scale ----
    # Small trees stay compact in the middle (short edges); the biggest tree
    # fills the frame. Node radius is derived from the same scale so circles
    # never clip the edge or overlap, regardless of tree size.
    max_span_x = max(s["extent"][1] - s["extent"][0] for s in stages)
    max_span_y = max(s["extent"][3] - s["extent"][2] for s in stages)
    MARGIN = 0.13                       # keep node centres clear of the frame edge
    usable = 1.0 - 2.0 * MARGIN
    fits = [usable / span for span in (max_span_x, max_span_y) if span > 0]
    SPREAD = 1.2          # <<< line length: >1 = longer edges, <1 = shorter
    scale = (min(fits) if fits else usable) * SPREAD
    node_r = 0.075        # <<< circle radius, now independent of line length

    def place(stage):
        x0, x1, y0, y1 = stage["extent"]
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

        def put(n):
            n.x = 0.5 + (n.x - cx) * scale
            n.y = 0.5 + (n.y - cy) * scale
            for c in n.children:
                put(c)

        put(stage["tree"])

    for s in stages:
        place(s)
    # ---- frame schedule: (stage_index, alpha) ----
    frames = []
    for k in range(len(stages)):
        if k == 0:
            for a in np.linspace(0, 1, args.morph):
                frames.append((0, a, 0))     # grow from truth-ish (prev = truth)
        else:
            for a in np.linspace(0, 1, args.morph):
                frames.append((k, a, k - 1))
        for _ in range(args.hold):
            frames.append((k, 1.0, k))
    # extra hold on the selected model at the end
    sel_k = next((i for i, s in enumerate(stages) if s["selected"]), len(stages) - 1)
    for _ in range(args.fps * 2):
        frames.append((sel_k, 1.0, sel_k))

    # ---- figure ----
    fig = plt.figure(figsize=(13, 7.0), facecolor=BG)
    ax_t = fig.add_axes([0.04, 0.06, 0.40, 0.68])          # tree (smaller, centred)
    ax_t.set_facecolor(BG)
    ax_t.axis("off")
    ax3 = fig.add_axes([0.52, 0.05, 0.43, 0.78], projection="3d")
    ax3.set_facecolor(BG)

    sup = fig.text(0.5, 0.92, "", ha="center", va="center", color=FG, fontsize=30)

    zmin = float(min(gamma_truth.min(), min(s["gamma_pred"].min() for s in stages)))
    zmax = float(max(gamma_truth.max(), max(s["gamma_pred"].max() for s in stages)))
    zpad = 0.05 * (zmax - zmin)

    def draw_tree(stage, highlight):
        ax_t.clear()
        ax_t.axis("off")
        ax_t.set_xlim(-0.08, 1.08)
        ax_t.set_ylim(-0.12, 1.12)
        ax_t.set_aspect("equal", adjustable="box")
        ax_t.apply_aspect()   # finalise transform so we can measure text vs circle
        root = stage["tree"]
        # selected model: edges in the predicted-marker colour
        ec = PRED_C if highlight else NODE_EC
        nodes = []

        def collect(n):
            nodes.append(n)
            for c in n.children:
                collect(c)

        collect(root)
        # one shared radius for every tree (computed once, above); the label
        # auto-fit below shrinks long glyphs to sit inside the circle.
        r = node_r
        base_fs = 26.0

        # edges first
        def draw_edges(n):
            for c in n.children:
                ax_t.plot([n.x, c.x], [n.y, c.y], color=ec, lw=1, alpha=1, zorder=1)
                draw_edges(c)

        draw_edges(root)
        # circle diameter in display pixels (for auto-fitting the label)
        R = fig.canvas.get_renderer()
        p0 = ax_t.transData.transform((0.0, 0.0))
        p1 = ax_t.transData.transform((2.0 * r, 0.0))
        dia_px = abs(p1[0] - p0[0])
        for n in nodes:
            circ = plt.Circle((n.x, n.y), r, facecolor=NODE_FC, edgecolor=ec,
                              lw=1.8, zorder=2)
            ax_t.add_patch(circ)
            t = ax_t.text(n.x, n.y, n.label, ha="center", va="center",
                          color=FG, fontsize=base_fs, zorder=3)
            # shrink so the glyphs sit inside the circle (~80% of diameter)
            bb = t.get_window_extent(renderer=R)
            if bb.width > 0 and bb.height > 0:
                scale = min(0.80 * dia_px / bb.width, 0.80 * dia_px / bb.height, 1.0)
                if scale < 1.0:
                    t.set_fontsize(max(6.0, base_fs * scale))

    def update(fi):
        k, alpha, kprev = frames[fi]
        st = stages[k]
        prev_pred = gamma_truth if kprev == k and k == 0 else stages[kprev]["gamma_pred"]
        if k == 0:
            # grow prediction in from a flat plane at the data mean
            base = np.full_like(gamma_truth, gamma_truth.mean())
            cur = base * (1 - alpha) + st["gamma_pred"] * alpha
        else:
            cur = prev_pred * (1 - alpha) + st["gamma_pred"] * alpha

        highlight = st["selected"] and alpha >= 1.0
        draw_tree(st, highlight)

        ax3.clear()
        ax3.set_facecolor(BG)
        ax3.scatter(T_K, P_bar, gamma_truth, s=20, c=TRUTH_C, alpha=0.7,
                    depthshade=True, label="Truth")
        ax3.scatter(T_K, P_bar, cur, s=20, c=PRED_C, alpha=0.7,
                    depthshade=True, label="Prediction")
        ax3.set_xlabel(r"$T\ /\ [\mathrm{K}]$", color=FG, labelpad=12, fontsize=22)
        ax3.set_ylabel(r"$P\ /\ [\mathrm{bar}]$", color=FG, labelpad=12, fontsize=22)
        ax3.set_zlabel(r"$\gamma\ /\ [\mathrm{mN/m}]$", color=FG, labelpad=12, fontsize=22)
        ax3.set_zlim(zmin - zpad, zmax + zpad)
        ax3.tick_params(colors=FG, labelsize=16, pad=2)
        for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            pane.set_pane_color((0, 0, 0, 0.03))
        ax3.view_init(elev=22, azim=-60)
        leg = ax3.legend(loc="upper right", fontsize=18, framealpha=0.0, borderpad=0.5)
        for txt in leg.get_texts():
            txt.set_color(FG)

        sup.set_text(st["title"])
        # selected model: equation in the predicted-marker colour
        sup.set_color(PRED_C if highlight else FG)
        return []

    return dict(fig=fig, update=update, frames=frames, stem=stem,
                n_stages=len(stages), sel_complexity=sel_complexity)


# Built once in the parent; forked render workers inherit it (copy-on-write),
# so we never pickle the figure/closure or re-read the data per worker.
_SCENE = None


def _render_chunk(indices, tmpdir, dpi):
    """Render a subset of frame indices to PNGs (runs in a forked worker)."""
    fig = _SCENE["fig"]
    update = _SCENE["update"]
    for fi in indices:
        update(fi)
        fig.savefig(os.path.join(tmpdir, f"frame_{fi:05d}.png"),
                    dpi=dpi, facecolor=BG)
    return len(indices)


def encode_outputs(args, stem, tmpdir, n_frames):
    want_gif = args.format in ("both", "gif")
    want_mp4 = args.format in ("both", "mp4")
    pattern = os.path.join(tmpdir, "frame_%05d.png")

    if (want_mp4 or want_gif) and _HAVE_FFMPEG:
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        common = [ff, "-y", "-hide_banner", "-loglevel", "error"]
        if want_mp4:
            out_mp4 = str(stem.with_suffix(".mp4"))
            subprocess.run(common + [
                "-framerate", str(args.fps), "-i", pattern,
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                out_mp4], check=True)
            print(f"[{args.label}] wrote {out_mp4}  ({n_frames} frames @ {args.fps} fps)")
        if want_gif:
            out_gif = str(stem.with_suffix(".gif"))
            palette = os.path.join(tmpdir, "palette.png")
            subprocess.run(common + [
                "-framerate", str(args.fps), "-i", pattern,
                "-vf", "palettegen", palette], check=True)
            subprocess.run(common + [
                "-framerate", str(args.fps), "-i", pattern, "-i", palette,
                "-lavfi", "paletteuse", out_gif], check=True)
            print(f"[{args.label}] wrote {out_gif}  ({n_frames} frames @ {args.fps} fps)")
        return

    # no ffmpeg: Pillow GIF fallback; MP4 not possible
    if want_mp4:
        print(f"[{args.label}] imageio-ffmpeg not available; skipping MP4")
    if want_gif:
        from PIL import Image
        out_gif = str(stem.with_suffix(".gif"))
        paths = [os.path.join(tmpdir, f"frame_{i:05d}.png") for i in range(n_frames)]
        imgs = [Image.open(p).convert("RGB") for p in paths]
        imgs[0].save(out_gif, save_all=True, append_images=imgs[1:],
                     duration=int(round(1000.0 / args.fps)), loop=0, disposal=2)
        print(f"[{args.label}] wrote {out_gif}  ({n_frames} frames @ {args.fps} fps)")


def run(args):
    global _SCENE
    sc = build_scene(args)
    _SCENE = sc
    n_frames = len(sc["frames"])
    stem = sc["stem"]
    print(f"[{args.label}] {sc['n_stages']} equations, "
          f"selected complexity = {sc['sel_complexity']}")

    # warm the font cache + renderer in the parent so forks inherit them,
    # and fail fast (clear traceback) if a single frame can't be drawn.
    sc["update"](0)
    sc["fig"].canvas.draw()

    jobs = args.jobs if args.jobs and args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, n_frames))

    tmpdir = tempfile.mkdtemp(prefix=f"srframes_{args.label}_")
    t0 = time.time()
    try:
        if jobs == 1:
            _render_chunk(list(range(n_frames)), tmpdir, args.dpi)
        else:
            # round-robin: spread the heavier (larger-tree) frames across workers
            chunks = [list(range(i, n_frames, jobs)) for i in range(jobs)]
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
                futs = [ex.submit(_render_chunk, ch, tmpdir, args.dpi)
                        for ch in chunks if ch]
                for fut in futs:
                    fut.result()
        print(f"[{args.label}] rendered {n_frames} frames on {jobs} worker(s) "
              f"in {time.time() - t0:.1f}s (dpi={args.dpi})")
        encode_outputs(args, stem, tmpdir, n_frames)
        if args.keep_frames:
            framedir = stem.parent / f"{stem.name}_frames"
            if framedir.exists():
                shutil.rmtree(framedir, ignore_errors=True)
            framedir.mkdir(parents=True, exist_ok=True)
            for i in range(n_frames):
                name = f"frame_{i:05d}.png"
                shutil.copy2(os.path.join(tmpdir, name), framedir / name)
            print(f"[{args.label}] saved {n_frames} frames to {framedir}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
