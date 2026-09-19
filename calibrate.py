from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path

import yaml
from loguru import logger

from coords import CoordBook, CoordObject
from runtime import GameWindowCapture, make_game_capture, poll_escape


TOGGLE_KEY = "F10"
HIDE_PAD = 8
HIDE_W, HIDE_H = 92, 28
POINT_R = 7


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("config.yaml должен быть словарём")
    return data


def _poll_f10() -> bool:
    if sys.platform == "darwin":
        from human_input_mac import _key_down

        return _key_down(109)
    import win32api
    import win32con

    return bool(win32api.GetAsyncKeyState(win32con.VK_F10) & 0x8000)


def _prepare_mac_overlay(root: tk.Tk) -> None:
    if sys.platform != "darwin":
        return
    try:
        from AppKit import (
            NSApplication,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowCollectionBehaviorStationary,
        )

        app = NSApplication.sharedApplication()
        behavior = (
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
        )
        for window in app.windows():
            window.setCollectionBehavior_(behavior)
            window.setLevel_(1000)
            window.setIgnoresMouseEvents_(False)
    except Exception:
        logger.exception("Не удалось вывести оверлей на все Spaces macOS")


class Calibrator:
    def __init__(self, capture: GameWindowCapture, book: CoordBook) -> None:
        self.capture = capture
        self.book = book
        self.selected = min(book.objects)
        self.visible = True
        self._digit_buffer = ""
        self._digit_at = 0.0
        self._f10_was = False
        self.root = tk.Tk()
        self.root.title("Miscrits coords")
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
        self.root.after(40, self._tick)
        self.root.after(200, lambda: _prepare_mac_overlay(self.root))

    def _object(self) -> CoordObject:
        return self.book.objects[self.selected]

    def _sync_window(self) -> None:
        region = self.capture.get_region()
        self.capture.last_region = region
        self.root.geometry(f"{region.width}x{region.height}+{region.left}+{region.top}")
        self.canvas.config(width=region.width, height=region.height)

    def _draw(self) -> None:
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

    def _dot(self, x: int, y: int, color: str, label: str) -> None:
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
        if self._hit_hide(event.x, event.y):
            self._set_visible(False)
            return
        point = self._local(event)
        if point is None:
            return
        obj = self._object()
        if obj.kind == "pixel":
            self.book.set_pixel(obj.number, point[0], point[1])
        else:
            self.book.set_corner(obj.number, "tl", point[0], point[1])
        self._draw()

    def _on_right(self, event: tk.Event) -> None:
        point = self._local(event)
        if point is None:
            return
        obj = self._object()
        if obj.kind == "rect":
            self.book.set_corner(obj.number, "br", point[0], point[1])
            self._draw()

    @staticmethod
    def _hit_hide(x: int, y: int) -> bool:
        return HIDE_PAD <= x <= HIDE_PAD + HIDE_W and HIDE_PAD <= y <= HIDE_PAD + HIDE_H

    def _on_key(self, event: tk.Event) -> None:
        key = event.keysym
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
        if visible:
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self.root.focus_force()
            _prepare_mac_overlay(self.root)
        else:
            self.root.withdraw()
            logger.info("Оверлей скрыт. {} — показать.", TOGGLE_KEY)

    def _tick(self) -> None:
        if poll_escape():
            self.root.quit()
            return
        f10 = _poll_f10()
        if f10 and not self._f10_was:
            self._set_visible(not self.visible)
        self._f10_was = f10
        if self.visible:
            try:
                self._sync_window()
                self._draw()
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
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Калибровка координат поверх окна Miscrits")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>")
    config = _load_config(Path(args.config))
    capture = make_game_capture(config)
    try:
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
        Calibrator(capture, book).run()
        return 0
    finally:
        capture.close()


if __name__ == "__main__":
    raise SystemExit(main())
