from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FrameLayout:
    """Пересчёт координат эталона 1920×1080 в пиксели текущего снимка и на экран."""

    ref_w: int
    ref_h: int
    frame_w: int
    frame_h: int
    window_left: int
    window_top: int
    window_width: int
    window_height: int
    fit: str
    content_left: float
    content_top: float
    content_w: float
    content_h: float
    match: float = 1.0

    @classmethod
    def build(
        cls,
        frame_w: int,
        frame_h: int,
        window_left: int,
        window_top: int,
        window_width: int,
        window_height: int,
        ref_w: int = 1920,
        ref_h: int = 1080,
        fit: str = "fill",
        match: float = 1.0,
    ) -> FrameLayout:
        frame_w = max(1, int(frame_w))
        frame_h = max(1, int(frame_h))
        ref_w = max(1, int(ref_w))
        ref_h = max(1, int(ref_h))
        mode = str(fit or "fill").strip().lower()
        blend = min(max(float(match), 0.0), 1.0)
        # Maximized Windows client is often 1920×1009: title bar + taskbar
        # eat height, but Miscrits still paints UI 1:1 from the top-left.
        # Identity only when the frame is shorter — fullscreen is taller and
        # the game uniformly scales that maximized UI (cover / crop sides).
        chrome_slack = 140
        width_match = abs(frame_w - ref_w) <= 2
        shorter_chrome = width_match and 0 <= (ref_h - frame_h) <= chrome_slack
        if mode in {"none", "identity"} or (mode == "fill" and shorter_chrome):
            content_left = 0.0
            content_top = 0.0
            content_w = float(ref_w)
            content_h = float(ref_h)
            mode = "identity"
        elif mode == "contain":
            scale = min(frame_w / ref_w, frame_h / ref_h)
            content_w = ref_w * scale
            content_h = ref_h * scale
            content_left = (frame_w - content_w) / 2.0
            content_top = (frame_h - content_h) / 2.0
        elif mode == "cover":
            fill_sx = frame_w / ref_w
            fill_sy = frame_h / ref_h
            cover_s = max(fill_sx, fill_sy)
            scale_x = fill_sx + blend * (cover_s - fill_sx)
            scale_y = fill_sy + blend * (cover_s - fill_sy)
            content_w = ref_w * scale_x
            content_h = ref_h * scale_y
            cover_left = (frame_w - ref_w * cover_s) / 2.0
            cover_top = (frame_h - ref_h * cover_s) / 2.0
            content_left = blend * cover_left
            content_top = blend * cover_top
        else:
            content_left = 0.0
            content_top = 0.0
            content_w = float(frame_w)
            content_h = float(frame_h)
            mode = "fill"
        return cls(
            ref_w=ref_w,
            ref_h=ref_h,
            frame_w=frame_w,
            frame_h=frame_h,
            window_left=int(window_left),
            window_top=int(window_top),
            window_width=max(1, int(window_width)),
            window_height=max(1, int(window_height)),
            fit=mode,
            content_left=content_left,
            content_top=content_top,
            content_w=content_w,
            content_h=content_h,
            match=blend,
        )

    @classmethod
    def from_frame(
        cls,
        frame_w: int,
        frame_h: int,
        ref_w: int = 1920,
        ref_h: int = 1080,
        fit: str = "fill",
        match: float = 1.0,
    ) -> FrameLayout:
        return cls.build(
            frame_w,
            frame_h,
            0,
            0,
            frame_w,
            frame_h,
            ref_w=ref_w,
            ref_h=ref_h,
            fit=fit,
            match=match,
        )

    @property
    def scale_x(self) -> float:
        return self.content_w / self.ref_w

    @property
    def scale_y(self) -> float:
        return self.content_h / self.ref_h

    def is_identity(self) -> bool:
        return (
            abs(self.scale_x - 1.0) < 0.004
            and abs(self.scale_y - 1.0) < 0.004
            and abs(self.content_left) < 1.0
            and abs(self.content_top) < 1.0
        )

    def ref_to_frame(self, x: float, y: float) -> tuple[int, int]:
        fx = self.content_left + float(x) * self.scale_x
        fy = self.content_top + float(y) * self.scale_y
        return (
            int(min(max(round(fx), 0), self.frame_w - 1)),
            int(min(max(round(fy), 0), self.frame_h - 1)),
        )

    def ref_to_frame_rect(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> tuple[int, int, int, int]:
        x1, y1 = self.ref_to_frame(left, top)
        x2, y2 = self.ref_to_frame(right, bottom)
        if x2 <= x1:
            x2 = min(self.frame_w, x1 + 1)
        if y2 <= y1:
            y2 = min(self.frame_h, y1 + 1)
        return x1, y1, x2, y2

    def ref_to_frame_size(self, width: float, height: float) -> tuple[int, int]:
        return (
            max(1, int(round(float(width) * self.scale_x))),
            max(1, int(round(float(height) * self.scale_y))),
        )

    def frame_to_ref(self, x: float, y: float) -> tuple[int, int]:
        rx = (float(x) - self.content_left) / self.scale_x
        ry = (float(y) - self.content_top) / self.scale_y
        return int(round(rx)), int(round(ry))

    def frame_to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx = self.window_left + float(x) * self.window_width / self.frame_w
        sy = self.window_top + float(y) * self.window_height / self.frame_h
        return int(round(sx)), int(round(sy))

    def screen_to_frame(self, x: float, y: float) -> tuple[int, int]:
        fx = (float(x) - self.window_left) * self.frame_w / self.window_width
        fy = (float(y) - self.window_top) * self.frame_h / self.window_height
        return (
            int(min(max(round(fx), 0), self.frame_w - 1)),
            int(min(max(round(fy), 0), self.frame_h - 1)),
        )

    def ref_to_screen(self, x: float, y: float) -> tuple[int, int]:
        return self.frame_to_screen(*self.ref_to_frame(x, y))
