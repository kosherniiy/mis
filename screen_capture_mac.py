from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from time import sleep, time

import cv2
import mss
import numpy as np
from loguru import logger

try:
    import Quartz
    from AppKit import (
        NSApplicationActivateAllWindows,
        NSApplicationActivateIgnoringOtherApps,
        NSRunningApplication,
        NSWorkspace,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Для macOS нужны пакеты pyobjc-framework-Quartz и pyobjc-framework-Cocoa. "
        "Установи зависимости из requirements-mac.txt"
    ) from exc


@dataclass(frozen=True, slots=True)
class CaptureRegion:
    """Клиентская область окна в экранных координатах."""

    left: int
    top: int
    width: int
    height: int


def _window_value(window: object, *keys: object) -> object:
    for key in keys:
        if key is None:
            continue
        try:
            if key in window:  # type: ignore[operator]
                value = window[key]  # type: ignore[index]
                if value is not None:
                    return value
        except Exception:
            pass
        getter = getattr(window, "get", None)
        if callable(getter):
            try:
                value = getter(key)
                if value is not None:
                    return value
            except Exception:
                pass
    return None


def _window_name(window: object) -> str:
    return str(
        _window_value(window, Quartz.kCGWindowName, "kCGWindowName") or ""
    )


def _window_owner(window: object) -> str:
    return str(
        _window_value(window, Quartz.kCGWindowOwnerName, "kCGWindowOwnerName")
        or ""
    )


def _window_pid(window: object) -> int:
    raw = _window_value(window, Quartz.kCGWindowOwnerPID, "kCGWindowOwnerPID")
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _window_layer(window: object) -> int:
    raw = _window_value(window, Quartz.kCGWindowLayer, "kCGWindowLayer")
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _window_bounds(window: object) -> dict[str, float]:
    raw = _window_value(window, Quartz.kCGWindowBounds, "kCGWindowBounds") or {}
    try:
        return {
            "X": float(raw.get("X", 0)),
            "Y": float(raw.get("Y", 0)),
            "Width": float(raw.get("Width", 0)),
            "Height": float(raw.get("Height", 0)),
        }
    except Exception:
        return {"X": 0.0, "Y": 0.0, "Width": 0.0, "Height": 0.0}


def _window_matches(title: str, name: str, owner: str, extra_owner: str) -> bool:
    needle = title.casefold()
    extra = extra_owner.casefold().strip()
    haystacks = (name.casefold(), owner.casefold())
    if any(needle and needle in item for item in haystacks):
        return True
    return bool(extra and extra in owner.casefold())


def _list_windows() -> list[object]:
    # Без OnScreenOnly: иначе fullscreen на отдельном Space невидим.
    raw = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    return list(raw or [])


def _describe_windows(limit: int = 12) -> str:
    visible: list[str] = []
    for window in _list_windows():
        name = _window_name(window)
        owner = _window_owner(window)
        bounds = _window_bounds(window)
        layer = _window_layer(window)
        if not name and not owner:
            continue
        visible.append(
            f"{owner}: {name or '—'} "
            f"{int(bounds['Width'])}x{int(bounds['Height'])} layer={layer}"
        )
        if len(visible) >= limit:
            break
    return "; ".join(visible) or "нет доступных имён (нужен Screen Recording)"


def _osa_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _run_osascript(*lines: str) -> bool:
    try:
        completed = subprocess.run(
            ["osascript", *[item for line in lines for item in ("-e", line)]],
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )
        if completed.returncode == 0:
            return True
        err = (completed.stderr or "").strip()
        if err:
            logger.debug("osascript: {}", err)
        return False
    except Exception:
        logger.debug("osascript не выполнился")
        return False


def _running_app(pid: int, owner: str):
    if pid:
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if app is not None:
            return app
    workspace = NSWorkspace.sharedWorkspace()
    owner_cf = owner.casefold()
    for app in workspace.runningApplications():
        if pid and int(app.processIdentifier()) == pid:
            return app
        localized = str(app.localizedName() or "").casefold()
        if owner_cf and (owner_cf == localized or owner_cf in localized or localized in owner_cf):
            return app
    return None


def _activate_ns(app) -> bool:
    try:
        app.unhide()
    except Exception:
        pass
    try:
        if bool(app.isActive()):
            return True
    except Exception:
        pass
    options = NSApplicationActivateAllWindows | NSApplicationActivateIgnoringOtherApps
    try:
        activate = getattr(app, "activate", None)
        if callable(activate) and bool(activate()):
            return True
    except Exception:
        logger.debug("NSRunningApplication.activate() не сработал")
    try:
        current = NSRunningApplication.currentApplication()
        method = getattr(app, "activateFromApplication_options_error_", None)
        if callable(method) and current is not None:
            result = method(current, options, None)
            if result is True or result == (True, None) or (
                isinstance(result, tuple) and result and result[0]
            ):
                return True
    except Exception:
        logger.debug("activateFromApplication не сработал")
    try:
        if bool(app.activateWithOptions_(options)):
            return True
        app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        return True
    except Exception:
        logger.debug("activateWithOptions не сработал")
        return False


def _activate_osascript(pid: int, owner: str) -> bool:
    if pid:
        if _run_osascript(
            "tell application \"System Events\"",
            f"set frontmost of first process whose unix id is {pid} to true",
            "end tell",
        ):
            return True
    if not owner:
        return False
    escaped = _osa_escape(owner)
    if _run_osascript(f'tell application "{escaped}" to activate'):
        return True
    return _run_osascript(
        "tell application \"System Events\"",
        f'set frontmost of first process whose name is "{escaped}" to true',
        "end tell",
    )


class GameWindowCapture:
    """Захватывает клиентскую область окна игры на macOS."""

    def __init__(
        self,
        window_title: str,
        titlebar_height: int = 0,
        window_owner: str = "",
        **_unused: object,
    ) -> None:
        self.window_title = window_title
        self.titlebar_height = max(0, int(titlebar_height))
        self.window_owner = window_owner
        self._sct = mss.mss()
        self.last_region: CaptureRegion | None = None
        self.last_capture_at = 0.0
        self._last_owner = ""
        self._logged_size = False

    def _find_window(self) -> object:
        own_pid = os.getpid()
        named: list[tuple[object, float]] = []
        owned: list[tuple[object, float]] = []
        title = self.window_title.casefold()
        for window in _list_windows():
            if _window_layer(window) != 0:
                continue
            if _window_pid(window) == own_pid:
                continue
            name = _window_name(window)
            owner = _window_owner(window)
            if "coords" in name.casefold():
                continue
            if not _window_matches(self.window_title, name, owner, self.window_owner):
                continue
            bounds = _window_bounds(window)
            width = bounds["Width"]
            height = bounds["Height"]
            if width < 200 or height < 200:
                continue
            # CGWindowList часто отдаёт фантом 500x500@(0,400) без имени.
            if not name and int(width) == 500 and int(height) == 500:
                continue
            item = (window, width * height)
            if title and title in name.casefold():
                named.append(item)
            else:
                owned.append(item)
        pool = named or owned
        if not pool:
            raise RuntimeError(
                f"Окно с заголовком '{self.window_title}' не найдено, "
                f"включая другие рабочие столы. Видно: {_describe_windows()}"
            )
        return max(pool, key=lambda item: item[1])[0]

    def focus(self) -> bool:
        """Переводит процесс игры на передний план, в том числе с другого Space."""
        try:
            window = self._find_window()
            owner = _window_owner(window)
            pid = _window_pid(window)
            self._last_owner = owner
            app = _running_app(pid, owner)
            already_front = False
            try:
                already_front = bool(app is not None and app.isActive())
            except Exception:
                already_front = False
            activated = False
            if app is not None:
                activated = _activate_ns(app)
            if not activated:
                activated = _activate_osascript(pid, owner)
            if not activated and app is None:
                logger.error(
                    "Процесс окна '{}' pid={} не найден для фокуса. Видно: {}",
                    owner,
                    pid,
                    _describe_windows(),
                )
                return False
            if not activated:
                logger.warning(
                    "macOS не подтвердил activate для '{}' pid={}; "
                    "пробую продолжить после переключения Space",
                    owner,
                    pid,
                )
            if not already_front:
                # Анимация отдельного fullscreen-стола.
                sleep(0.55)
            logger.info("Окно игры на переднем плане: {} pid={}", owner or self.window_title, pid)
            return True
        except RuntimeError as exc:
            logger.error("{}", exc)
            return False
        except Exception:
            logger.exception("Не удалось сфокусировать окно игры")
            return False

    def get_region(self) -> CaptureRegion:
        window = self._find_window()
        bounds = _window_bounds(window)
        left = int(round(bounds["X"]))
        top = int(round(bounds["Y"]))
        width = int(round(bounds["Width"]))
        height = int(round(bounds["Height"]))
        top += self.titlebar_height
        height -= self.titlebar_height
        if width <= 0 or height <= 0:
            raise RuntimeError("Клиентская область окна имеет нулевой размер")
        self._last_owner = _window_owner(window)
        return CaptureRegion(left, top, width, height)

    def capture(self) -> np.ndarray:
        """Возвращает снимок клиентской области в формате BGR, в логических пикселях."""
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
            if frame.shape[1] != region.width or frame.shape[0] != region.height:
                frame = cv2.resize(
                    frame,
                    (region.width, region.height),
                    interpolation=cv2.INTER_AREA,
                )
            self.last_region = region
            self.last_capture_at = time()
            if not self._logged_size:
                self._logged_size = True
                logger.info(
                    "Захват окна: логический {}x{} @ ({}, {}), снимок {}x{}",
                    region.width,
                    region.height,
                    region.left,
                    region.top,
                    frame.shape[1],
                    frame.shape[0],
                )
            return frame
        except Exception:
            logger.exception("Ошибка захвата окна игры")
            raise

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        region = self.last_region
        if region is None:
            region = self.get_region()
        return region.left + int(x), region.top + int(y)

    def from_screen(self, x: int, y: int) -> tuple[int, int]:
        region = self.last_region
        if region is None:
            region = self.get_region()
        return int(x) - region.left, int(y) - region.top

    def contains_point(self, x: int, y: int) -> bool:
        try:
            region = self.last_region or self.get_region()
        except Exception:
            return False
        return (
            region.left <= x < region.left + region.width
            and region.top <= y < region.top + region.height
        )

    def close(self) -> None:
        self._sct.close()
