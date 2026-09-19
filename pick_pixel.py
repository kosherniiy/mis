from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from runtime import (
    cursor_screen_position,
    make_game_capture,
    poll_escape,
    poll_left_mouse,
)


WINDOW_NAME = "Выбор пикселя — ЛКМ: координаты, ESC/Q: выход"


def format_pixel(x: int, y: int, blue: int, green: int, red: int) -> str:
    hex_color = f"#{red:02X}{green:02X}{blue:02X}"
    return (
        f"x={x}, y={y} | RGB=({red}, {green}, {blue}) | "
        f"BGR=({blue}, {green}, {red}) | HEX={hex_color}"
    )


def run_global_picker() -> int:
    """Отслеживает ЛКМ над игрой и выдаёт координаты клиентской области."""
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    capture = make_game_capture(config)
    result_path = Path("selected_pixel.txt")
    was_pressed = False

    print("Кликните ЛКМ по нужной точке прямо в окне игры.")
    print("Результат появится здесь и в selected_pixel.txt. ESC — выход.")
    try:
        while True:
            if poll_escape():
                return 0

            is_pressed = poll_left_mouse()
            if is_pressed and not was_pressed:
                screen_x, screen_y = cursor_screen_position()
                region = capture.get_region()
                local_x = screen_x - region.left
                local_y = screen_y - region.top

                if 0 <= local_x < region.width and 0 <= local_y < region.height:
                    frame = capture.capture()
                    blue, green, red = (int(value) for value in frame[local_y, local_x])
                    result = format_pixel(local_x, local_y, blue, green, red)
                    print(result, flush=True)
                    result_path.write_text(result + "\n", encoding="utf-8")
                    print(f"Сохранено: {result_path.resolve()}", flush=True)
                    return 0

                print("Клик был вне клиентской области окна Miscrits.", flush=True)

            was_pressed = is_pressed
            time.sleep(0.01)
    finally:
        capture.close()


def run_image_picker(path: Path) -> int:
    """Показывает файл в отдельном окне и обрабатывает клики по нему."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Не удалось открыть изображение: {path}")

    display = image.copy()

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        nonlocal display
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        blue, green, red = (int(value) for value in image[y, x])
        hex_color = f"#{red:02X}{green:02X}{blue:02X}"
        result = format_pixel(x, y, blue, green, red)
        print(result, flush=True)
        Path("selected_pixel.txt").write_text(result + "\n", encoding="utf-8")

        display = image.copy()
        cv2.circle(display, (x, y), 8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(display, (x - 13, y), (x + 13, y), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(display, (x, y - 13), (x, y + 13), (0, 0, 255), 1, cv2.LINE_AA)
        label = f"({x}, {y}) {hex_color}"
        cv2.putText(
            display,
            label,
            (max(4, x + 12), max(20, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    cv2.imshow(WINDOW_NAME, display)

    print("Нажмите ЛКМ на нужном пикселе. Для выхода нажмите ESC или Q.")
    while True:
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Показывает координаты и цвет пикселя при клике ЛКМ."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="PNG/JPG для анализа. Без аргумента отслеживается клик по окну игры.",
    )
    args = parser.parse_args()
    if args.image is None:
        return run_global_picker()
    return run_image_picker(args.image)


if __name__ == "__main__":
    raise SystemExit(main())
