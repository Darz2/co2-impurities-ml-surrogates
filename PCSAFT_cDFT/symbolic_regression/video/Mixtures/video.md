# SR discovery video — how to render

`uv` reads the script's inline dependency block, so there's nothing to install.
`--no-project` makes uv ignore the A6 workspace and build a small throwaway env.

Frames are rendered to PNGs in parallel across all CPU cores and encoded once
with ffmpeg, so a full re-render takes a few seconds.

```bash
cd /home/darshan/A6/PCSAFT_cDFT/symbolic_regression/video/Mixtures

# Version 1
uv run --no-project sr_tree_video.py --outdir SR_MIXTURES_OUTPUTS --label V1

# Version 2 (borrows T/P from V1's run since V2's CSV has no Tr/Pr)
uv run --no-project sr_tree_video.py --outdir SR_MIXTURES_OUTPUTS_V2 --label V2 \
    --tp-from SR_MIXTURES_OUTPUTS --nsample 600
```

Outputs land here as `SR_tree_video_V1.{gif,mp4}` and `SR_tree_video_V2.{gif,mp4}`.

## Useful flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--jobs N` | all CPUs | parallel render workers (`--jobs 1` = single process) |
| `--dpi N` | 120 | frame resolution (raise for sharper / larger files) |
| `--format` | both | `gif`, `mp4`, or `both` |
| `--fps N` | 10 | playback frame rate (lower = slower) |
| `--morph N` | 10 | frames to morph between equations |
| `--hold N` | 8 | frames to hold on each equation |
| `--nsample N` | 700 | points shown in the 3D cloud |
| `--keep-frames` | off | also save the individual PNG frames to `<stem>_frames/` |
| `--out PATH` | — | explicit output stem (extension sets format) |

See everything with:

```bash
uv run --no-project sr_tree_video.py --help
```
