"""The hidden `synth` command: write one synthetic test clip, for checking an installation by hand.

`cli/main.py` adds the command with `add_parser`. It is not listed in `--help`. The clips are those of
outline_tracker/synthetic/clips.py. Only argparse and the standard library are imported here: OpenCV, pandas and
scikit-image are loaded when the command runs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from outline_tracker.cli.errors import report_error


def add_parser(commands) -> argparse.ArgumentParser:
    """Add `synth` to the subcommands of `outline-tracker` (the object that
    `ArgumentParser.add_subparsers` returns) and return its parser. No quantities, so no units."""
    # Hidden: a subcommand added without `help=` is not listed in `outline-tracker --help`.
    synth = commands.add_parser(
        "synth",
        description="Write a synthetic test clip (1920 x 1080 px, 240 fps, H.264) with known contents.",
    )
    synth.add_argument("scene", choices=["dish", "closeup"],
                       help="dish: small shrimp in a dish, a contact and an LED (32.4 um/px); "
                            "closeup: one shrimp beating its antennae at 9 Hz (10 um/px)")
    synth.add_argument("out", metavar="OUT.mp4", help="the clip to write; its folder is created if needed")
    synth.add_argument("--seconds", type=float, default=2.0, metavar="S",
                       help="length of the clip in s (default: 2)")
    synth.set_defaults(run=run)
    return synth


def run(args: argparse.Namespace) -> int:
    """`synth {dish,closeup} OUT.mp4 [--seconds S]`: write a synthetic clip and say what it shows.

    The clip is 1920 x 1080 px at 240 fps (also its fps_true) and S seconds long. The printed
    positions are in px in Tracker's convention (SPEC 3.1), frame numbers count from 0. Returns 1
    with an `ERROR:` line, and writes nothing, if S is not a positive number.
    """
    if not (math.isfinite(args.seconds) and args.seconds > 0):
        report_error(ValueError(f"--seconds must be a positive number of seconds, not {args.seconds:g}."))
        return 1
    from outline_tracker.synthetic import clips as synthetic  # imported here: it loads OpenCV, pandas and scikit-image

    make = {"dish": synthetic.dish_scene, "closeup": synthetic.closeup_scene}[args.scene]
    scene = make(n_frames=max(int(round(args.seconds * synthetic.FPS)), 1))
    out = Path(args.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        synthetic.render(scene, out)
    except (OSError, RuntimeError, ValueError) as error:
        report_error(error)
        return 1
    print(f"wrote {out}: {scene.n_frames} frames, {scene.size[0]} x {scene.size[1]} px, {scene.fps:g} fps")
    names = ", ".join(obj.track_id for obj in scene.objects)
    print(f"  scale {scene.mm_per_px:g} mm/px; fps_true {scene.fps:g}; objects {names}")
    if scene.dish is not None:
        print("  dish wall: center ({:.1f}, {:.1f}) px, radius {:.1f} px".format(*scene.dish))
    if scene.led is not None:
        print("  LED box u0,v0,u1,v1 = {},{},{},{} px switches on at frame {}".format(
            *scene.led.box_px, scene.led.onset_frame))
    if scene.contact is not None:
        print("  {} and {} pass {:g} px apart at frame {}".format(
            *scene.contact.track_ids, scene.contact.gap_px, scene.contact.frame))
    return 0
