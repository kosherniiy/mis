from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger

from hp_ocr import read_enemy_hp
from runtime import make_game_capture

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEBUG_ROOT = Path("debug") / "hp_ocr"


def load_config(path: str | Path = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Конфиг не найден: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Корень config.yaml должен быть YAML-объектом")
    return config


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Файл пуст: {path}")
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return frame


def collect_image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"В папке нет скриншотов: {path}")
        return files
    raise FileNotFoundError(f"Скриншот не найден: {path}")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", value, flags=re.UNICODE).strip("._")
    return cleaned[:80] or "image"


def configure_file_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        log_path,
        level="DEBUG",
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    )


def silence_logger() -> None:
    logger.remove()


def print_hp(
    frame: np.ndarray,
    config: dict,
    *,
    label: str | None,
    debug_dir: Path | None,
) -> int:
    result = read_enemy_hp(
        frame,
        config,
        debug_dir=debug_dir,
        debug_label=label,
    )
    prefix = f"{label}: " if label else ""
    suffix = f"  [отладка: {debug_dir}]" if debug_dir is not None else ""
    if result is None:
        print(f"{prefix}HP не распознан{suffix}")
        return 1
    current_hp, total_hp = result
    print(f"{prefix}{current_hp}/{total_hp}{suffix}")
    return 0


def read_once(config: dict, run_dir: Path | None, index: int = 1) -> int:
    capture = make_game_capture(config)
    try:
        frame = capture.capture()
    except Exception as exc:
        print(f"Не удалось снять окно игры: {exc}", file=sys.stderr)
        logger.exception("Не удалось снять окно игры")
        return 1
    finally:
        capture.close()
    debug_dir = run_dir / f"{index:03d}_live" if run_dir is not None else None
    return print_hp(frame, config, label="live", debug_dir=debug_dir)


def read_images(config: dict, image_path: Path, run_dir: Path | None) -> int:
    try:
        paths = collect_image_paths(image_path)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        logger.exception("Не удалось собрать скриншоты")
        return 2

    failed = 0
    many = len(paths) > 1
    for index, path in enumerate(paths, start=1):
        try:
            frame = load_image(path)
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            logger.exception("Не удалось прочитать {}", path)
            failed += 1
            continue
        debug_dir = (
            run_dir / f"{index:03d}_{safe_name(path.stem)}"
            if run_dir is not None
            else None
        )
        label = str(path) if many else path.name
        failed += print_hp(frame, config, label=label, debug_dir=debug_dir)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Читает HP мискрита той же OCR-проверкой, что и бот.",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=Path,
        metavar="PATH",
        help="Скриншот или папка со скриншотами вместо захвата окна игры.",
    )
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SEC",
        help="Повторять проверку каждые SEC секунд.",
    )
    parser.add_argument(
        "-d",
        "--d",
        action="store_true",
        dest="debug",
        help="Сохранять отладочные скрины и логи OCR HP в debug/hp_ocr.",
    )
    args = parser.parse_args()

    try:
        config = load_config()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Ошибка загрузки config.yaml: {exc}", file=sys.stderr)
        return 2

    if args.image is not None and args.watch is not None:
        print("Нельзя одновременно использовать --image и --watch.", file=sys.stderr)
        return 2

    run_dir: Path | None = None
    if args.debug:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        run_dir = DEBUG_ROOT / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        configure_file_logging(run_dir / "session.log")
        logger.info("Запуск проверки HP, отладка: {}", run_dir.resolve())
        print(f"Отладка OCR HP: {run_dir}")
    else:
        silence_logger()

    if args.image is not None:
        return read_images(config, args.image, run_dir)

    if args.watch is None:
        return read_once(config, run_dir)

    interval = max(0.2, args.watch)
    index = 1
    try:
        while True:
            read_once(config, run_dir, index)
            index += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
