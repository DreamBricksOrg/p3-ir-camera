#!/usr/bin/env python3
"""TouchDesigner bridge for P3/P1 thermal cameras on Windows.

This script captures frames from the camera and publishes them via Spout so
TouchDesigner can ingest them with a Spout In TOP.

Requirements (Windows):
  pip install SpoutGL

Usage:
  python touchdesigner_spout_bridge.py --sender P3Thermal
  python touchdesigner_spout_bridge.py --model p1 --sender P1Thermal --fps 30

TouchDesigner:
  1. Add "Spout In TOP"
  2. Set Sender Name to match --sender (default: P3Thermal)
  3. Use Resolution from sender (auto)
"""

from __future__ import annotations

import argparse
import contextlib
import time

from OpenGL import GL

import cv2
import numpy as np

from p3_camera import Model, P3Camera, get_model_config, raw_to_celsius_corrected
from p3_viewer import agc_fixed, agc_temporal, apply_colormap, parse_probe_pixel


def parse_args() -> argparse.Namespace:
    def _parse_probe_pixel_arg(val: str) -> tuple[int, int]:
        try:
            return parse_probe_pixel(val)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    parser = argparse.ArgumentParser(
        description="Publish P3/P1 thermal video to TouchDesigner via Spout",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="p1",
        choices=["p1", "p3"],
        help="Camera model",
    )
    parser.add_argument(
        "--sender",
        type=str,
        default="P1Thermal",
        help="Spout sender name shown in TouchDesigner",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Target output FPS",
    )
    parser.add_argument(
        "--agc",
        type=str,
        default="temporal",
        choices=["factory", "temporal", "fixed"],
        help="AGC mode before colormap",
    )
    parser.add_argument(
        "--colormap",
        type=int,
        default=0,
        help=(
            "Colormap ID from p3_viewer.ColormapID "
            "(0=WHITE_HOT, 1=BLACK_HOT, 2=RAINBOW, 3=IRONBOW, 4=MILITARY, 5=SEPIA)"
        ),
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Integer upscaling for output image",
    )
    parser.add_argument(
        "--osc-host",
        type=str,
        default="127.0.0.1",
        help="OSC destination host for frame_stats metadata",
    )
    parser.add_argument(
        "--osc-port",
        type=int,
        default=9000,
        help="OSC destination port for frame_stats metadata",
    )
    parser.add_argument(
        "--osc-prefix",
        type=str,
        default="/p3",
        help="OSC address prefix (e.g. /p3 sends /p3/tspot, /p3/tmin, ...)",
    )
    parser.add_argument(
        "--no-osc",
        action="store_true",
        help="Disable OSC metadata output",
    )
    parser.add_argument(
        "--probe-pixel",
        type=_parse_probe_pixel_arg,
        default=None,
        metavar="X,Y",
        help=(
            "Sample corrected temperature at sensor-space pixel X,Y and include "
            "it in OSC frame_stats output"
        ),
    )
    return parser.parse_args()


def to_display_u8(ir_brightness: np.ndarray, thermal_raw: np.ndarray, agc: str) -> np.ndarray:
    if agc == "factory":
        return ir_brightness
    if agc == "fixed":
        return agc_fixed(thermal_raw)
    return agc_temporal(thermal_raw, pct=1.0)


def build_frame_stats(
    thermal_raw: np.ndarray,
    display_u8: np.ndarray,
    env: object,
    probe_pixel: tuple[int, int] | None = None,
) -> dict[str, float | int]:
    """Create a metadata dictionary compatible with p3_viewer frame statistics."""
    h, w = thermal_raw.shape
    cy, cx = h // 2, w // 2
    cmin = int(thermal_raw.argmin())
    cmax = int(thermal_raw.argmax())
    min_y, min_x = divmod(cmin, w)
    max_y, max_x = divmod(cmax, w)

    frame_stats: dict[str, float | int] = {
        "tspot": float(raw_to_celsius_corrected(thermal_raw[cy, cx], env)),
        "cmin": cmin,
        "cmax": cmax,
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "tmin": float(raw_to_celsius_corrected(thermal_raw.ravel()[cmin], env)),
        "tmax": float(raw_to_celsius_corrected(thermal_raw.ravel()[cmax], env)),
    }

    if probe_pixel is not None:
        px, py = probe_pixel
        frame_stats["probe_x"] = px
        frame_stats["probe_y"] = py
        frame_stats["tprobe"] = float(raw_to_celsius_corrected(thermal_raw[py, px], env))

    return frame_stats


def send_frame_stats_osc(osc_client: object, prefix: str, frame_stats: dict[str, float | int]) -> None:
    base = f"/{prefix.strip('/')}"
    osc_client.send_message(f"{base}/tspot", float(frame_stats["tspot"]))
    osc_client.send_message(f"{base}/tmin", float(frame_stats["tmin"]))
    osc_client.send_message(f"{base}/tmax", float(frame_stats["tmax"]))
    osc_client.send_message(f"{base}/cmin", int(frame_stats["cmin"]))
    osc_client.send_message(f"{base}/cmax", int(frame_stats["cmax"]))
    osc_client.send_message(f"{base}/min_x", int(frame_stats["min_x"]))
    osc_client.send_message(f"{base}/min_y", int(frame_stats["min_y"]))
    osc_client.send_message(f"{base}/max_x", int(frame_stats["max_x"]))
    osc_client.send_message(f"{base}/max_y", int(frame_stats["max_y"]))
    if "tprobe" in frame_stats:
        osc_client.send_message(f"{base}/tprobe", float(frame_stats["tprobe"]))
        osc_client.send_message(f"{base}/probe_x", int(frame_stats["probe_x"]))
        osc_client.send_message(f"{base}/probe_y", int(frame_stats["probe_y"]))


def main() -> int:
    args = parse_args()

    try:
        import SpoutGL  # type: ignore[import-not-found]
    except ImportError:
        print("ERROR: SpoutGL is not installed.")
        print("Install it on Windows with: pip install SpoutGL")
        return 1

    osc_client = None
    if not args.no_osc:
        try:
            from pythonosc.udp_client import (
                SimpleUDPClient,  # type: ignore[import-not-found]
            )
        except ImportError:
            print("WARNING: python-osc is not installed; continuing without OSC metadata output.")
            print("Install it with: pip install python-osc")
        else:
            osc_client = SimpleUDPClient(args.osc_host, args.osc_port)

    config = get_model_config(Model(args.model))
    if args.probe_pixel is not None:
        px, py = args.probe_pixel
        if px >= config.sensor_w or py >= config.sensor_h:
            raise SystemExit(
                f"--probe-pixel out of range for {config.model.value}: "
                f"valid x=0..{config.sensor_w - 1}, y=0..{config.sensor_h - 1}"
            )

    camera = P3Camera(config=config)

    frame_period = 1.0 / max(args.fps, 1.0)

    sender = SpoutGL.SpoutSender()  # pyright: ignore[reportAttributeAccessIssue]
    sender.setSenderName(args.sender)  # pyright: ignore[reportAttributeAccessIssue]

    try:
        camera.connect()
        camera.init()
        camera.start_streaming()

        print(f"Connected to {args.model.upper()} camera")
        print(f"Spout sender: {args.sender}")
        if osc_client is not None:
            print(
                f"OSC metadata -> {args.osc_host}:{args.osc_port} "
                f"(prefix: /{args.osc_prefix.strip('/')})"
            )
        print("Press Ctrl+C to stop")

        while True:
            t0 = time.perf_counter()
            ir_brightness, thermal_raw = camera.read_frame_both()

            if ir_brightness is None or thermal_raw is None:
                continue

            gray = to_display_u8(ir_brightness, thermal_raw, args.agc)
            frame_stats = build_frame_stats(
                thermal_raw,
                gray,
                camera.env_params,
                probe_pixel=args.probe_pixel,
            )
            bgr = apply_colormap(gray, args.colormap)

            if args.scale > 1:
                h, w = bgr.shape[:2]
                bgr = cv2.resize(
                    bgr,
                    (w * args.scale, h * args.scale),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Spout expects RGBA in many Python wrappers
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
            h, w = rgba.shape[:2]

            sender.sendImage(rgba, w, h, GL.GL_RGBA, False, 0)  # pyright: ignore[reportAttributeAccessIssue]
            sender.setFrameSync(args.sender)  # pyright: ignore[reportAttributeAccessIssue]
            if osc_client is not None:
                send_frame_stats_osc(osc_client, args.osc_prefix, frame_stats)

            elapsed = time.perf_counter() - t0
            sleep_s = frame_period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nStopping bridge...")
    finally:
        with contextlib.suppress(Exception):
            camera.stop_streaming()
        with contextlib.suppress(Exception):
            camera.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
