from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from coords import CoordBook, CoordObject
from runtime import GameWindowCapture, make_game_capture, poll_escape


TOGGLE_KEY = "F10"
HIDE_PAD = 8
HIDE_W, HIDE_H = 92, 28
POINT_R = 7

# HID keycodes for overlay hotkeys when the overlay cannot become key.
_MAC_F10 = 109
_MAC_KEY_S = 1
_MAC_KEY_Q = 12
_MAC_BRACKET_LEFT = 33
_MAC_BRACKET_RIGHT = 30
_MAC_DIGITS = {
    18: "1",
    19: "2",
    20: "3",
    21: "4",
    23: "5",
    22: "6",
    26: "7",
    28: "8",
    25: "9",
    29: "0",
}

_MAC_TYPES: dict[str, Any] | None = None


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("config.yaml должен быть словарём")
    return data


def _poll_f10() -> bool:
    if sys.platform == "darwin":
        from human_input_mac import _key_down

        return _key_down(_MAC_F10)
    import win32api
    import win32con

    return bool(win32api.GetAsyncKeyState(win32con.VK_F10) & 0x8000)


def _hex_color(value: str, alpha: float = 1.0):
    from AppKit import NSColor

    raw = value.lstrip("#")
    red = int(raw[0:2], 16) / 255.0
    green = int(raw[2:4], 16) / 255.0
    blue = int(raw[4:6], 16) / 255.0
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(red, green, blue, alpha)


def _cocoa_rect(left: int, top: int, width: int, height: int):
    from AppKit import NSScreen
    from Foundation import NSMakeRect

    primary = NSScreen.screens()[0]
    cocoa_y = primary.frame().size.height - float(top) - float(height)
    return NSMakeRect(float(left), cocoa_y, float(width), float(height))


def _mac_overlay_types() -> dict[str, Any]:
    """NSPanel, не NSWindow: FullScreenAuxiliary на чужом fullscreen-Space иначе игнорируется."""
    global _MAC_TYPES
    if _MAC_TYPES is not None:
        return _MAC_TYPES

    from AppKit import NSPanel, NSView
    from Foundation import NSMakePoint, NSMakeRect, NSString

    class OverlayPanel(NSPanel):
        def canBecomeKeyWindow(self) -> bool:  # noqa: N802
            return False

        def canBecomeMainWindow(self) -> bool:  # noqa: N802
            return False

    class OverlayView(NSView):
        calibrator = None

        def isFlipped(self) -> bool:  # noqa: N802
            return True

        def isOpaque(self) -> bool:  # noqa: N802
            return False

        def acceptsFirstMouse_(self, _event) -> bool:  # noqa: N802
            return True

        def mouseDown_(self, event) -> None:  # noqa: N802
            owner = self.calibrator
            if owner is None:
                return
            point = self.convertPoint_fromView_(event.locationInWindow(), None)
            owner._click_left(int(point.x), int(point.y))

        def rightMouseDown_(self, event) -> None:  # noqa: N802
            owner = self.calibrator
            if owner is None:
                return
            point = self.convertPoint_fromView_(event.locationInWindow(), None)
            owner._click_right(int(point.x), int(point.y))

        def drawRect_(self, _rect) -> None:  # noqa: N802
            owner = self.calibrator
            if owner is not None:
                owner._paint_mac(self)

    _MAC_TYPES = {
        "panel": OverlayPanel,
        "view": OverlayView,
        "NSMakePoint": NSMakePoint,
        "NSMakeRect": NSMakeRect,
        "NSString": NSString,
    }
    return _MAC_TYPES


class MacOverlay:
    """Нативная панель поверх fullscreen-стола другого приложения."""

    def __init__(self, calibrator: Calibrator) -> None:
        from AppKit import (
            NSApplication,
            NSApplicationActivationPolicyAccessory,
            NSBackingStoreBuffered,
            NSColor,
            NSViewHeightSizable,
            NSViewWidthSizable,
            NSWindowAnimationBehaviorNone,
            NSWindowCollectionBehaviorCanJoinAllApplications,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowCollectionBehaviorIgnoresCycle,
            NSWindowCollectionBehaviorStationary,
            NSWindowStyleMaskBorderless,
            NSWindowStyleMaskNonactivatingPanel,
        )
        from Quartz import CGWindowLevelForKey, kCGAssistiveTechHighWindowLevelKey

        types = _mac_overlay_types()
        app = NSApplication.sharedApplication()
        try:
            app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        except Exception:
            pass

        self.calibrator = calibrator
        self._behavior = (
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorCanJoinAllApplications
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorIgnoresCycle
        )
        self._nonactivating = NSWindowStyleMaskNonactivatingPanel
        self._level = int(CGWindowLevelForKey(kCGAssistiveTechHighWindowLevelKey))
        region = calibrator.capture.last_region
        frame = _cocoa_rect(
            region.left if region else 0,
            region.top if region else 0,
            region.width if region else 400,
            region.height if region else 300,
        )
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = types["panel"].alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            style,
            NSBackingStoreBuffered,
            False,
        )
        panel.setTitle_("Coords overlay")
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(False)
        panel.setReleasedWhenClosed_(False)
        panel.setFloatingPanel_(True)
        panel.setBecomesKeyOnlyIfNeeded_(True)
        panel.setAnimationBehavior_(NSWindowAnimationBehaviorNone)
        view = types["view"].alloc().initWithFrame_(panel.contentView().bounds())
        view.calibrator = calibrator
        view.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        panel.setContentView_(view)
        self.panel = panel
        self.view = view
        self._last_frame = frame
        self._apply_space_flags()
        panel.orderFrontRegardless()

    def _apply_space_flags(self) -> None:
        panel = self.panel
        panel.setCollectionBehavior_(self._behavior)
        panel.setLevel_(self._level)
        panel.setHidesOnDeactivate_(False)
        panel.setIgnoresMouseEvents_(False)
        panel.setStyleMask_(panel.styleMask() | self._nonactivating)

    def set_region(self, left: int, top: int, width: int, height: int) -> None:
        frame = _cocoa_rect(left, top, width, height)
        if (
            abs(frame.origin.x - self._last_frame.origin.x) < 0.5
            and abs(frame.origin.y - self._last_frame.origin.y) < 0.5
            and abs(frame.size.width - self._last_frame.size.width) < 0.5
            and abs(frame.size.height - self._last_frame.size.height) < 0.5
        ):
            return
        self._last_frame = frame
        self.panel.setFrame_display_(frame, True)
        self._apply_space_flags()
        self.panel.orderFrontRegardless()

    def set_visible(self, visible: bool) -> None:
        if visible:
            self._apply_space_flags()
            self.panel.orderFrontRegardless()
            self.redraw()
        else:
            self.panel.orderOut_(None)

    def redraw(self) -> None:
        self.view.setNeedsDisplay_(True)

    def keep_front(self) -> None:
        if int(self.panel.level()) < self._level:
            self._apply_space_flags()
        self.panel.orderFrontRegardless()

    def close(self) -> None:
        try:
            self.panel.orderOut_(None)
            self.panel.close()
        except Exception:
            pass


class Calibrator:
    def __init__(self, capture: GameWindowCapture, book: CoordBook) -> None:
        self.capture = capture
        self.book = book
        self.selected = min(book.objects)
        self.visible = True
        self._digit_buffer = ""
        self._digit_at = 0.0
        self._f10_was = _poll_f10()
        self._mac_keys_was: dict[int, bool] = {}
        self._last_geometry = ""
        self._mac: MacOverlay | None = None
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Coords overlay")
        if sys.platform == "darwin":
            self.root.geometry("1x1+-2000+-2000")
            self.canvas = None
            self._mac = MacOverlay(self)
        else:
            self.root.overrideredirect(True)
            self.root.attributes("-topmost", True)
            try:
                self.root.attributes("-alpha", 0.42)
            except tk.TclError:
                pass
            self.canvas = tk.Canvas(self.root, highlightthickness=0, bg="#101820")
            self.canvas.pack(fill=tk.BOTH, expand=True)
            self.canvas.bind("<Button-1>", self._on_left)
            self.canvas.bind("<Button-3>", self._on_right)
            self.root.bind("<Key>", self._on_key)
            region = capture.last_region
            if region is not None:
                self._apply_geometry(region.width, region.height, region.left, region.top)
            self.root.deiconify()
        self.root.after(40, self._tick)

    def _object(self) -> CoordObject:
        return self.book.objects[self.selected]

    def _apply_geometry(self, width: int, height: int, left: int, top: int) -> None:
        geometry = f"{width}x{height}+{left}+{top}"
        if geometry == self._last_geometry:
            return
        self._last_geometry = geometry
        if self._mac is not None:
            self._mac.set_region(left, top, width, height)
            return
        assert self.canvas is not None
        self.root.geometry(geometry)
        self.canvas.config(width=width, height=height)

    def _sync_window(self) -> None:
        region = self.capture.get_region()
        self.capture.last_region = region
        self._apply_geometry(region.width, region.height, region.left, region.top)

    def _draw(self) -> None:
        if self._mac is not None:
            self._mac.redraw()
            return
        assert self.canvas is not None
        self.canvas.delete("all")
        region = self.capture.last_region
        if region is None:
            return
        obj = self._object()
        if obj.kind == "pixel":
            self._dot(obj.x, obj.y, "#ff3b3b", f"{obj.number} {obj.name}")
        else:
            self.canvas.create_rectangle(
                obj.left,
                obj.top,
                obj.right,
                obj.bottom,
                outline="#39ff14",
                width=2,
            )
            self._dot(obj.left, obj.top, "#4da3ff", f"{obj.number} {obj.name} · ЛВ")
            self._dot(obj.right, obj.bottom, "#ffb347", "ПН")
        self.canvas.create_rectangle(
            HIDE_PAD,
            HIDE_PAD,
            HIDE_PAD + HIDE_W,
            HIDE_PAD + HIDE_H,
            fill="#1c2a3a",
            outline="#9ad1ff",
        )
        self.canvas.create_text(
            HIDE_PAD + HIDE_W / 2,
            HIDE_PAD + HIDE_H / 2,
            text="Скрыть",
            fill="#ffffff",
            font=("Segoe UI", 10, "bold"),
        )
        hint = (
            f"#{obj.number} {obj.name}  [{obj.kind}]   "
            f"{self.book.size_key}   цифры=номер  [/]=листы  S=сохранить  "
            f"{TOGGLE_KEY}=показать/скрыть"
        )
        self.canvas.create_rectangle(8, region.height - 32, region.width - 8, region.height - 8, fill="#1c2a3a", outline="")
        self.canvas.create_text(
            region.width / 2,
            region.height - 20,
            text=hint,
            fill="#ffffff",
            font=("Segoe UI", 10),
        )

    def _paint_mac(self, view) -> None:
        from AppKit import NSBezierPath, NSFont, NSFontAttributeName, NSForegroundColorAttributeName

        types = _mac_overlay_types()
        region = self.capture.last_region
        bounds = view.bounds()
        _hex_color("#101820", 0.45).set()
        NSBezierPath.fillRect_(bounds)
        if region is None:
            return
        obj = self._object()
        if obj.kind == "pixel":
            self._mac_dot(types, obj.x, obj.y, "#ff3b3b", f"{obj.number} {obj.name}")
        else:
            _hex_color("#39ff14").set()
            path = NSBezierPath.bezierPathWithRect_(
                types["NSMakeRect"](
                    float(obj.left),
                    float(obj.top),
                    float(max(1, obj.right - obj.left)),
                    float(max(1, obj.bottom - obj.top)),
                )
            )
            path.setLineWidth_(2.0)
            path.stroke()
            self._mac_dot(types, obj.left, obj.top, "#4da3ff", f"{obj.number} {obj.name} · ЛВ")
            self._mac_dot(types, obj.right, obj.bottom, "#ffb347", "ПН")
        _hex_color("#1c2a3a").set()
        NSBezierPath.fillRect_(
            types["NSMakeRect"](float(HIDE_PAD), float(HIDE_PAD), float(HIDE_W), float(HIDE_H))
        )
        _hex_color("#9ad1ff").set()
        border = NSBezierPath.bezierPathWithRect_(
            types["NSMakeRect"](float(HIDE_PAD), float(HIDE_PAD), float(HIDE_W), float(HIDE_H))
        )
        border.setLineWidth_(1.0)
        border.stroke()
        self._mac_text(
            types,
            "Скрыть",
            HIDE_PAD + HIDE_W / 2,
            HIDE_PAD + HIDE_H / 2 - 7,
            "#ffffff",
            12,
            center=True,
        )
        hint = (
            f"#{obj.number} {obj.name}  [{obj.kind}]   "
            f"{self.book.size_key}   цифры=номер  [/]=листы  S=сохранить  "
            f"{TOGGLE_KEY}=показать/скрыть"
        )
        _hex_color("#1c2a3a").set()
        NSBezierPath.fillRect_(
            types["NSMakeRect"](8.0, float(region.height - 32), float(region.width - 16), 24.0)
        )
        self._mac_text(types, hint, region.width / 2, region.height - 27, "#ffffff", 11, center=True)
        _ = NSFont, NSFontAttributeName, NSForegroundColorAttributeName

    def _mac_dot(self, types: dict[str, Any], x: int, y: int, color: str, label: str) -> None:
        from AppKit import NSBezierPath

        _hex_color(color).set()
        oval = NSBezierPath.bezierPathWithOvalInRect_(
            types["NSMakeRect"](float(x - POINT_R), float(y - POINT_R), float(POINT_R * 2), float(POINT_R * 2))
        )
        oval.fill()
        self._mac_text(types, label, x + 12, y - 16, color, 12)

    def _mac_text(
        self,
        types: dict[str, Any],
        text: str,
        x: float,
        y: float,
        color: str,
        size: int,
        center: bool = False,
    ) -> None:
        from AppKit import NSFont, NSFontAttributeName, NSForegroundColorAttributeName

        attrs = {
            NSFontAttributeName: NSFont.boldSystemFontOfSize_(float(size)),
            NSForegroundColorAttributeName: _hex_color(color),
        }
        ns_text = types["NSString"].stringWithString_(text)
        point_x = float(x)
        if center:
            drawn = ns_text.sizeWithAttributes_(attrs)
            point_x -= drawn.width / 2.0
        ns_text.drawAtPoint_withAttributes_(types["NSMakePoint"](point_x, float(y)), attrs)

    def _dot(self, x: int, y: int, color: str, label: str) -> None:
        assert self.canvas is not None
        self.canvas.create_oval(
            x - POINT_R,
            y - POINT_R,
            x + POINT_R,
            y + POINT_R,
            outline=color,
            width=2,
            fill=color,
        )
        self.canvas.create_text(x + 12, y - 12, text=label, fill=color, anchor="w", font=("Segoe UI", 10, "bold"))

    def _local(self, event: tk.Event) -> tuple[int, int] | None:
        region = self.capture.last_region
        if region is None:
            return None
        x, y = int(event.x), int(event.y)
        if not (0 <= x < region.width and 0 <= y < region.height):
            return None
        return x, y

    def _on_left(self, event: tk.Event) -> None:
        self._click_left(int(event.x), int(event.y))

    def _on_right(self, event: tk.Event) -> None:
        self._click_right(int(event.x), int(event.y))

    def _click_left(self, x: int, y: int) -> None:
        if self._hit_hide(x, y):
            self._set_visible(False)
            return
        region = self.capture.last_region
        if region is None or not (0 <= x < region.width and 0 <= y < region.height):
            return
        obj = self._object()
        if obj.kind == "pixel":
            self.book.set_pixel(obj.number, x, y)
        else:
            self.book.set_corner(obj.number, "tl", x, y)
        self._draw()

    def _click_right(self, x: int, y: int) -> None:
        region = self.capture.last_region
        if region is None or not (0 <= x < region.width and 0 <= y < region.height):
            return
        obj = self._object()
        if obj.kind == "rect":
            self.book.set_corner(obj.number, "br", x, y)
            self._draw()

    @staticmethod
    def _hit_hide(x: int, y: int) -> bool:
        return HIDE_PAD <= x <= HIDE_PAD + HIDE_W and HIDE_PAD <= y <= HIDE_PAD + HIDE_H

    def _on_key(self, event: tk.Event) -> None:
        self._handle_keysym(event.keysym)

    def _handle_keysym(self, key: str) -> None:
        if key in {"Escape", "q", "Q"}:
            self.root.quit()
            return
        if key in {"s", "S"}:
            self.book.save()
            return
        if key == "bracketleft":
            self._select_delta(-1)
            return
        if key == "bracketright":
            self._select_delta(1)
            return
        if key.isdigit():
            now = time.monotonic()
            if now - self._digit_at > 0.7:
                self._digit_buffer = ""
            self._digit_buffer += key
            self._digit_at = now
            number = int(self._digit_buffer)
            if number in self.book.objects:
                self.selected = number
                self._draw()
            elif len(self._digit_buffer) >= 2:
                self._digit_buffer = key
                number = int(self._digit_buffer)
                if number in self.book.objects:
                    self.selected = number
                    self._draw()

    def _select_delta(self, delta: int) -> None:
        numbers = sorted(self.book.objects)
        index = numbers.index(self.selected)
        self.selected = numbers[(index + delta) % len(numbers)]
        self._draw()

    def _set_visible(self, visible: bool) -> None:
        self.visible = visible
        if self._mac is not None:
            self._mac.set_visible(visible)
            if not visible:
                logger.info("Оверлей скрыт. {} — показать.", TOGGLE_KEY)
            return
        if visible:
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self.root.update_idletasks()
        else:
            self.root.withdraw()
            logger.info("Оверлей скрыт. {} — показать.", TOGGLE_KEY)

    def _mac_edge(self, code: int, down: bool) -> bool:
        was = self._mac_keys_was.get(code, False)
        self._mac_keys_was[code] = down
        return down and not was

    def _poll_mac_hotkeys(self) -> bool:
        """Глобальные клавиши: панель не становится key, чтобы не выкинуть из fullscreen."""
        if sys.platform != "darwin" or not self.visible:
            return False
        from human_input_mac import _key_down

        if self._mac_edge(_MAC_KEY_Q, _key_down(_MAC_KEY_Q)):
            self.root.quit()
            return True
        if self._mac_edge(_MAC_KEY_S, _key_down(_MAC_KEY_S)):
            self.book.save()
        if self._mac_edge(_MAC_BRACKET_LEFT, _key_down(_MAC_BRACKET_LEFT)):
            self._select_delta(-1)
        if self._mac_edge(_MAC_BRACKET_RIGHT, _key_down(_MAC_BRACKET_RIGHT)):
            self._select_delta(1)
        for code, key in _MAC_DIGITS.items():
            if self._mac_edge(code, _key_down(code)):
                self._handle_keysym(key)
        return False

    def _tick(self) -> None:
        if poll_escape():
            self.root.quit()
            return
        if self._poll_mac_hotkeys():
            return
        f10 = _poll_f10()
        if f10 and not self._f10_was:
            self._set_visible(not self.visible)
        self._f10_was = f10
        if self.visible:
            try:
                self._sync_window()
                self._draw()
                if self._mac is not None:
                    self._mac.keep_front()
            except Exception:
                logger.exception("Не удалось привязать оверлей к окну игры")
        self.root.after(40, self._tick)

    def run(self) -> None:
        logger.info(
            "Калибровка {}. Объект {}, ЛКМ/ПКМ двигают точки, S сохраняет, {} прячет.",
            self.book.size_key,
            self.selected,
            TOGGLE_KEY,
        )
        try:
            self.root.mainloop()
        finally:
            if self._mac is not None:
                self._mac.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Калибровка координат поверх окна Miscrits")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>")
    config = _load_config(Path(args.config))
    capture = make_game_capture(config)
    try:
        try:
            region = capture.get_region()
        except Exception:
            if not capture.focus():
                logger.error("Не удалось сфокусировать окно игры")
                return 1
            time.sleep(0.3)
            region = capture.get_region()
        capture.last_region = region
        book = CoordBook(Path(str(config.get("coords_dir", "./layouts"))))
        try:
            book.load(region.width, region.height)
        except FileNotFoundError:
            logger.warning(
                "Файла для {}x{} нет. Копирую 1920x1009 как заготовку — поправь точки.",
                region.width,
                region.height,
            )
            template = CoordBook(Path(str(config.get("coords_dir", "./layouts"))))
            template.load(1920, 1009)
            book.objects = template.objects
            book.width, book.height = region.width, region.height
            book.path = book.directory / f"{region.width}x{region.height}.yaml"
            book.save()
        calibrator = Calibrator(capture, book)
        capture.focus()
        if calibrator._mac is not None:
            calibrator._mac.set_visible(True)
        logger.info("Оверлей {}x{} @ ({}, {})", region.width, region.height, region.left, region.top)
        calibrator.run()
        return 0
    finally:
        capture.close()


if __name__ == "__main__":
    raise SystemExit(main())
