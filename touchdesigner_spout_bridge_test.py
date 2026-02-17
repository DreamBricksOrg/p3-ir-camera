"""Unit tests for touchdesigner_spout_bridge.py."""

from __future__ import annotations

import numpy as np

from touchdesigner_spout_bridge import build_frame_stats, parse_args, send_frame_stats_osc


class DummyOscClient:
    def __init__(self) -> None:
        self.messages: list[tuple[str, float | int]] = []

    def send_message(self, path: str, value: float | int) -> None:
        self.messages.append((path, value))


def test_build_frame_stats_includes_probe_fields_when_configured(monkeypatch):
    monkeypatch.setattr(
        "touchdesigner_spout_bridge.raw_to_celsius_corrected",
        lambda raw, _env: float(raw) / 100.0,
    )
    thermal = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
    display = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    stats = build_frame_stats(thermal, display, env=object(), probe_pixel=(1, 0))

    assert stats["probe_x"] == 1
    assert stats["probe_y"] == 0
    assert stats["tprobe"] == 20.0


def test_send_frame_stats_osc_sends_probe_fields_when_present():
    client = DummyOscClient()
    stats: dict[str, float | int] = {
        "tspot": 10.0,
        "tmin": 8.0,
        "tmax": 12.0,
        "cmin": 1,
        "cmax": 2,
        "range_min": 3,
        "range_max": 4,
        "tprobe": 9.0,
        "probe_x": 5,
        "probe_y": 6,
    }

    send_frame_stats_osc(client, "p3", stats)

    paths = {path for path, _value in client.messages}
    assert "/p3/tprobe" in paths
    assert "/p3/probe_x" in paths
    assert "/p3/probe_y" in paths


def test_parse_args_accepts_probe_pixel(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["touchdesigner_spout_bridge.py", "--probe-pixel", "12,34"],
    )

    args = parse_args()

    assert args.probe_pixel == (12, 34)
