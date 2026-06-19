"""
StatsOverlay — paint live simulation stats onto RGBA frames.

Used by the door showcase scripts to draw a semi-transparent stats panel in
the top-right corner of each video frame: digital-twin location (building +
floor + room label), a green/red physics-state indicator, and the current
command angle of each tracked door.

Operates directly on HxWx{3,4} uint8 numpy arrays returned by
`isaacsim.sensors.camera.Camera.get_rgba()`. Call `draw(rgba, door_angles,
frame_idx)` each frame before piping to VideoRecorder.capture_rgba().
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


class StatsOverlay:
    """Paints a stats panel onto the top-right of each video frame.

    Args:
        location_lines: Lines describing digital-twin location (shown at top).
            e.g. ("MSOE Innovation Center", "Floor 2 · South Corridor").
        state_label:   Short label for the physics state, e.g. "OPERATIONAL".
        state_ok:      True → green indicator. False → red indicator.
        door_labels:   Short labels per door (e.g. "D1".."D8"). Used in the
                       live-angles block. Must have the same length as the
                       door_angles list passed to `draw()`.
        max_door_rows: Cap on how many doors to list (default 8).
        panel_width:   Width of the overlay panel in pixels (default 340).
        margin:        Margin from the frame edges (default 24).
    """

    # Colors (RGBA)
    PANEL_BG = (0, 0, 0, 180)
    PANEL_BORDER = (255, 255, 255, 90)
    HEADER = (255, 255, 255, 255)
    BODY = (220, 220, 220, 255)
    DIM = (160, 160, 160, 255)
    GREEN = (70, 220, 110, 255)
    RED = (240, 80, 80, 255)

    def __init__(
        self,
        location_lines: Sequence[str],
        state_label: str,
        state_ok: bool,
        door_labels: Sequence[str],
        *,
        max_door_rows: int = 8,
        panel_width: int = 340,
        margin: int = 24,
    ) -> None:
        self.location_lines = list(location_lines)
        self.state_label = state_label
        self.state_ok = state_ok
        self.door_labels = list(door_labels)
        self.max_door_rows = max_door_rows
        self.panel_width = panel_width
        self.margin = margin

        self._font_hdr = _load_font(20)
        self._font_body = _load_font(16)
        self._font_small = _load_font(13)

        # Measure header + static-row heights once; door rows are dynamic.
        self._row_h = 22
        self._hdr_row_h = 26

    # ---------------------------------------------------------------- drawing

    def draw(
        self,
        rgba: np.ndarray,
        door_angles_deg: Sequence[float],
        frame_idx: int = 0,
        total_frames: int | None = None,
    ) -> np.ndarray:
        """Return a new RGBA frame with the stats panel composited on top.

        Args:
            rgba:            HxWx{3,4} uint8 from camera.get_rgba().
            door_angles_deg: Current (commanded) angle per door in degrees.
            frame_idx:       Current frame number (for time readout).
            total_frames:    If given, shown as "frame X / total".
        """
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 3:
            return rgba

        base = rgba[:, :, :3].astype("uint8")
        H, W = base.shape[:2]

        img = Image.fromarray(base, mode="RGB").convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        shown = list(door_angles_deg)[: self.max_door_rows]
        extra = max(0, len(door_angles_deg) - len(shown))

        # Estimate panel height dynamically
        header_rows = len(self.location_lines)
        state_rows = 1
        doors_header_rows = 1
        door_rows = len(shown)
        extra_row = 1 if extra > 0 else 0
        time_row = 1

        pad_top = 14
        pad_bot = 14
        panel_h = (
            pad_top
            + self._hdr_row_h * header_rows
            + 8
            + self._row_h * state_rows
            + 10
            + self._row_h * doors_header_rows
            + self._row_h * door_rows
            + self._row_h * extra_row
            + 6
            + self._row_h * time_row
            + pad_bot
        )

        x1 = W - self.margin
        x0 = x1 - self.panel_width
        y0 = self.margin
        y1 = y0 + panel_h

        # Panel
        draw.rounded_rectangle(
            [(x0, y0), (x1, y1)], radius=10,
            fill=self.PANEL_BG, outline=self.PANEL_BORDER, width=1,
        )

        # --- Digital-twin location header ---
        cursor_y = y0 + pad_top
        text_x = x0 + 16
        for i, line in enumerate(self.location_lines):
            font = self._font_hdr if i == 0 else self._font_body
            color = self.HEADER if i == 0 else self.DIM
            draw.text((text_x, cursor_y), line, fill=color, font=font)
            cursor_y += self._hdr_row_h

        cursor_y += 4

        # Thin divider
        draw.line(
            [(x0 + 14, cursor_y), (x1 - 14, cursor_y)],
            fill=self.PANEL_BORDER, width=1,
        )
        cursor_y += 6

        # --- Physics state indicator ---
        dot_color = self.GREEN if self.state_ok else self.RED
        dot_r = 7
        dot_cx = text_x + dot_r
        dot_cy = cursor_y + self._row_h // 2
        draw.ellipse(
            [(dot_cx - dot_r, dot_cy - dot_r), (dot_cx + dot_r, dot_cy + dot_r)],
            fill=dot_color,
            outline=(255, 255, 255, 220), width=1,
        )
        draw.text(
            (dot_cx + dot_r + 10, cursor_y),
            f"Physics: {self.state_label}",
            fill=self.HEADER, font=self._font_body,
        )
        cursor_y += self._row_h + 6

        # Divider
        draw.line(
            [(x0 + 14, cursor_y), (x1 - 14, cursor_y)],
            fill=self.PANEL_BORDER, width=1,
        )
        cursor_y += 6

        # --- Door angles block ---
        draw.text(
            (text_x, cursor_y),
            f"Door Angles ({len(door_angles_deg)} active):",
            fill=self.HEADER, font=self._font_body,
        )
        cursor_y += self._row_h

        col2_x = x0 + self.panel_width // 2 + 12
        for i, angle in enumerate(shown):
            label = self.door_labels[i] if i < len(self.door_labels) else f"D{i+1}"
            # Left column: label
            draw.text(
                (text_x + 4, cursor_y),
                label,
                fill=self.DIM, font=self._font_body,
            )
            # Right column: angle (right-aligned-ish)
            angle_str = f"{angle:+7.1f}°"
            # Color angle text by magnitude — hot if heavily flailed
            if abs(angle) > 120.0:
                color = self.RED
            elif abs(angle) > 30.0:
                color = (255, 220, 120, 255)  # amber
            else:
                color = self.BODY
            draw.text(
                (col2_x, cursor_y),
                angle_str,
                fill=color, font=self._font_body,
            )
            cursor_y += self._row_h

        if extra > 0:
            draw.text(
                (text_x + 4, cursor_y),
                f"… (+{extra} more)",
                fill=self.DIM, font=self._font_small,
            )
            cursor_y += self._row_h

        cursor_y += 2
        # Divider
        draw.line(
            [(x0 + 14, cursor_y), (x1 - 14, cursor_y)],
            fill=self.PANEL_BORDER, width=1,
        )
        cursor_y += 4

        # --- Frame/time ---
        if total_frames is not None:
            t_str = f"frame {frame_idx:>4} / {total_frames}"
        else:
            t_str = f"frame {frame_idx}"
        draw.text(
            (text_x, cursor_y),
            t_str,
            fill=self.DIM, font=self._font_small,
        )

        composed = Image.alpha_composite(img, overlay).convert("RGB")
        out = np.asarray(composed, dtype=np.uint8)
        if rgba.shape[2] == 4:
            out = np.dstack([out, np.full(out.shape[:2], 255, dtype=np.uint8)])
        return out
