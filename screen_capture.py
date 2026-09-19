from __future__ import annotations

from dataclasses import dataclass
from time import time

import cv2
import mss
import numpy as np
import win32con
import win32gui
from loguru import logger


@dataclass(frozen=True, slots=True)
class CaptureRegion:
    """Клиентская область окна в экранных координатах."""

    left: int
    top: int
    width: int
    height: int


class GameWindowCapture:
    """Захватывает только клиентскую область окна игры."""

    def __init__(self, window_title: str, **_unused: object) -> None:
        self.window_title = window_title
        self._sct = mss.mss()
        self.last_region: CaptureRegion | None = None
        self.last_capture_at = 0.0

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
            self.last_region = region
            self.last_capture_at = time()
            return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        except Exception:
            logger.exception("Ошибка захвата окна игры")
            raise

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        """Переводит координаты снимка в абсолютные экранные координаты."""
        if self.last_region is None:
            raise RuntimeError("До преобразования координат необходимо сделать снимок")
        return self.last_region.left + x, self.last_region.top + y

    def from_screen(self, x: int, y: int) -> tuple[int, int]:
        """Переводит экранные координаты в координаты снимка."""
        if self.last_region is None:
            raise RuntimeError("До преобразования координат необходимо сделать снимок")
        return x - self.last_region.left, y - self.last_region.top

    def close(self) -> None:
        self._sct.close()
