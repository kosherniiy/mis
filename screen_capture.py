from __future__ import annotations

from dataclasses import dataclass
from time import time

import cv2
import mss
import numpy as np
import win32api
import win32con
import win32gui
from loguru import logger

from layout import FrameLayout


@dataclass(frozen=True, slots=True)
class CaptureRegion:
    """Клиентская область окна в экранных координатах."""

    left: int
    top: int
    width: int
    height: int


class GameWindowCapture:
    """Захватывает только клиентскую область окна игры."""

    def __init__(
        self,
        window_title: str,
        reference_width: int = 1920,
        reference_height: int = 1080,
        layout_fit: str = "fill",
        **_unused: object,
    ) -> None:
        self.window_title = window_title
        self.reference_width = max(1, int(reference_width))
        self.reference_height = max(1, int(reference_height))
        self.layout_fit = str(layout_fit or "fill")
        self._sct = mss.mss()
        self.last_region: CaptureRegion | None = None
        self.last_capture_at = 0.0
        self.layout = FrameLayout.from_frame(
            self.reference_width,
            self.reference_height,
            self.reference_width,
            self.reference_height,
            self.layout_fit,
        )
        self._logged_layout = False

    def _maximized_client_size(self) -> tuple[int, int] | None:
        """Клиент maximized-окна на текущем мониторе: рабочая область минус шапка."""
        try:
            hwnd = self._find_window()
            monitor = win32api.MonitorFromWindow(
                hwnd,
                win32con.MONITOR_DEFAULTTONEAREST,
            )
            info = win32api.GetMonitorInfo(monitor)
            work = info["Work"]
            work_w = int(work[2] - work[0])
            work_h = int(work[3] - work[1])
            caption = int(win32api.GetSystemMetrics(win32con.SM_CYCAPTION))
            return max(1, work_w), max(1, work_h - caption)
        except Exception:
            return None

    def _resolve_layout(
        self,
        region: CaptureRegion,
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int, str]:
        """Windowed 1920×1009 = identity; fullscreen scales that UI with cover."""
        requested = self.layout_fit
        maximized = self._maximized_client_size()
        if maximized is None:
            return self.reference_width, self.reference_height, requested
        max_w, max_h = maximized
        same_width = abs(region.width - max_w) <= 2
        if same_width and abs(region.height - max_h) <= 8:
            return frame_w, frame_h, "identity"
        if same_width and region.height >= max_h + 16:
            return max_w, max_h, "cover"
        return self.reference_width, self.reference_height, requested

    def _find_window(self) -> int:
        hwnd = win32gui.FindWindow(None, self.window_title)
        if not hwnd:
            candidates: list[tuple[int, str]] = []

            def collect(handle: int, _: object) -> None:
                title = win32gui.GetWindowText(handle)
                if win32gui.IsWindowVisible(handle) and self.window_title.lower() in title.lower():
                    candidates.append((handle, title))

            win32gui.EnumWindows(collect, None)
            if candidates:
                hwnd, matched_title = candidates[0]
                logger.debug("Окно найдено по части заголовка: {}", matched_title)
        if not hwnd:
            raise RuntimeError(f"Окно с заголовком '{self.window_title}' не найдено")
        return hwnd

    def focus(self) -> bool:
        """Восстанавливает и переводит окно игры на передний план."""
        try:
            hwnd = self._find_window()
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception:
            logger.exception("Не удалось сфокусировать окно игры")
            return False

    def get_region(self) -> CaptureRegion:
        hwnd = self._find_window()
        left, top = win32gui.ClientToScreen(hwnd, (0, 0))
        right, bottom = win32gui.ClientToScreen(
            hwnd,
            (win32gui.GetClientRect(hwnd)[2], win32gui.GetClientRect(hwnd)[3]),
        )
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            raise RuntimeError("Клиентская область окна имеет нулевой размер")
        return CaptureRegion(left, top, width, height)

    def capture(self) -> np.ndarray:
        """Возвращает снимок клиентской области в формате BGR."""
        try:
            region = self.get_region()
            raw = np.asarray(
                self._sct.grab(
                    {
                        "left": region.left,
                        "top": region.top,
                        "width": region.width,
                        "height": region.height,
                    }
                )
            )
            frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            self.last_region = region
            self.last_capture_at = time()
            ref_w, ref_h, fit = self._resolve_layout(
                region,
                frame.shape[1],
                frame.shape[0],
            )
            self.layout = FrameLayout.build(
                frame.shape[1],
                frame.shape[0],
                region.left,
                region.top,
                region.width,
                region.height,
                ref_w=ref_w,
                ref_h=ref_h,
                fit=fit,
            )
            if not self._logged_layout:
                self._logged_layout = True
                logger.info(
                    "Захват: окно {}x{} @ ({}, {}), снимок {}x{}, эталон {}x{}, "
                    "fit={}, масштаб {:.3f}x{:.3f}",
                    region.width,
                    region.height,
                    region.left,
                    region.top,
                    frame.shape[1],
                    frame.shape[0],
                    ref_w,
                    ref_h,
                    self.layout.fit,
                    self.layout.scale_x,
                    self.layout.scale_y,
                )
            return frame
        except Exception:
            logger.exception("Ошибка захвата окна игры")
            raise

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        """Переводит координаты снимка в абсолютные экранные координаты."""
        return self.layout.frame_to_screen(x, y)

    def from_screen(self, x: int, y: int) -> tuple[int, int]:
        """Переводит экранные координаты в координаты снимка."""
        return self.layout.screen_to_frame(x, y)

    def map_ref(self, x: int, y: int) -> tuple[int, int]:
        """Переводит точку из эталона 1920×1080 в пиксели текущего снимка."""
        return self.layout.ref_to_frame(x, y)

    def map_ref_rect(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> tuple[int, int, int, int]:
        return self.layout.ref_to_frame_rect(left, top, right, bottom)

    def close(self) -> None:
        self._sct.close()
