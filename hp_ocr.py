from __future__ import annotations

import re
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np
import pytesseract
from loguru import logger

from layout import FrameLayout
from tesseract_bin import resolve_tesseract_cmd

OCR_CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/"
OCR_CONFIG_NO_WHITELIST = "--psm 7 --oem 3"
HP_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)")
# Белый текст HP: низкая насыщенность, высокая яркость.
WHITE_TEXT_HSV_LOWER = (0, 0, 180)
WHITE_TEXT_HSV_UPPER = (180, 80, 255)
WHITE_TEXT_SCALE = 8
WHITE_TEXT_DILATE = 3


def tesseract_cmd_from_config(config: dict[str, Any]) -> str:
    return resolve_tesseract_cmd(
        str(config.get("capture_chance_area", {}).get("tesseract_cmd", "")).strip()
    )


def _write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"не удалось закодировать {path.name}")
    encoded.tofile(str(path))


def _clip_box(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    clipped_left = max(0, min(left, width))
    clipped_top = max(0, min(top, height))
    clipped_right = max(0, min(right, width))
    clipped_bottom = max(0, min(bottom, height))
    if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
        return None
    return clipped_left, clipped_top, clipped_right, clipped_bottom


def _image_stats(image: np.ndarray) -> str:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        channels = f"BGR mean={image.mean(axis=(0, 1)).round(2).tolist()}"
    else:
        gray = image
        channels = "gray"
    return (
        f"shape={image.shape} dtype={image.dtype} {channels} "
        f"min={int(gray.min())} max={int(gray.max())} "
        f"mean={float(gray.mean()):.2f} std={float(gray.std()):.2f}"
    )


def _ocr_dump(image: np.ndarray, config: str) -> dict[str, Any]:
    text = pytesseract.image_to_string(image, config=config)
    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    tokens: list[dict[str, Any]] = []
    n_boxes = len(data.get("text", []))
    for index in range(n_boxes):
        raw = str(data["text"][index])
        if not raw.strip():
            continue
        try:
            conf = float(data["conf"][index])
        except (TypeError, ValueError):
            conf = -1.0
        tokens.append(
            {
                "text": raw,
                "conf": conf,
                "left": int(data["left"][index]),
                "top": int(data["top"][index]),
                "width": int(data["width"][index]),
                "height": int(data["height"][index]),
            }
        )
    return {"text": text, "tokens": tokens}


def _write_ocr_result(log: TextIO, title: str, result: dict[str, Any]) -> None:
    text = result["text"]
    cleaned = text.replace(" ", "")
    match = HP_PATTERN.search(cleaned)
    log.write(f"\n=== {title} ===\n")
    log.write(f"raw={text!r}\n")
    log.write(f"stripped={cleaned!r}\n")
    if match is None:
        log.write("regex current/total: нет совпадения\n")
    else:
        log.write(f"regex current/total: {match.group(1)}/{match.group(2)}\n")
    tokens = result["tokens"]
    if not tokens:
        log.write("image_to_data: пусто\n")
        return
    log.write("image_to_data:\n")
    for token in tokens:
        log.write(
            "  text={text!r} conf={conf:.1f} box=({left},{top},"
            "{width}x{height})\n".format(**token)
        )


def _save_marked_frame(
    path: Path,
    frame: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    in_bounds: bool,
) -> None:
    marked = frame.copy()
    color = (0, 255, 0) if in_bounds else (0, 0, 255)
    height, width = marked.shape[:2]
    clipped = _clip_box(left, top, right, bottom, width, height)
    if clipped is not None:
        cv2.rectangle(
            marked,
            (clipped[0], clipped[1]),
            (max(clipped[0], clipped[2] - 1), max(clipped[1], clipped[3] - 1)),
            color,
            2,
        )
    label = f"HP ({left},{top})-({right},{bottom})"
    cv2.putText(
        marked,
        label,
        (max(8, left), max(24, top - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    _write_image(path, marked)


def _bright_text_mask(crop: np.ndarray, minimum: int) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, minimum, 255, cv2.THRESH_BINARY)
    return binary


def _white_text_mask(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.array(WHITE_TEXT_HSV_LOWER, dtype=np.uint8),
        np.array(WHITE_TEXT_HSV_UPPER, dtype=np.uint8),
    )


def _parse_hp_reading(text: str) -> tuple[int, int] | None:
    match = HP_PATTERN.search(text.replace(" ", ""))
    if match is None:
        return None
    current_hp = int(match.group(1))
    total_hp = int(match.group(2))
    if current_hp <= 0 or total_hp <= 0 or current_hp > total_hp:
        return None
    return current_hp, total_hp


def _three_nine_variant(left: int, right: int) -> bool:
    left_text = str(left)
    right_text = str(right)
    if len(left_text) != len(right_text):
        return False
    for left_digit, right_digit in zip(left_text, right_text):
        if left_digit == right_digit:
            continue
        if {left_digit, right_digit} != {"3", "9"}:
            return False
    return True


def _bar_looks_full(crop: np.ndarray) -> bool:
    height, width = crop.shape[:2]
    if height < 4 or width < 8:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    band = hsv[height // 3 : 2 * height // 3, 2 : max(3, int(width * 0.42))]
    if band.size == 0:
        return False
    saturation = band[:, :, 1]
    value = band[:, :, 2]
    bright_fill = float(((saturation > 80) & (value > 140)).mean())
    return bright_fill >= 0.4


def _pick_hp_reading(
    votes: list[tuple[int, int]],
    *,
    bar_full: bool = False,
) -> tuple[int, int] | None:
    if not votes:
        return None
    ranked = list(votes)
    if bar_full:
        for current_hp, total_hp in votes:
            if current_hp != total_hp and _three_nine_variant(current_hp, total_hp):
                ranked.append((total_hp, total_hp))
    counts = Counter(ranked)
    return max(
        counts,
        key=lambda reading: (
            counts[reading],
            int(bar_full and reading[0] == reading[1]),
            len(str(reading[0])) + len(str(reading[1])),
            reading[1],
            reading[0],
        ),
    )


def _scale_white_mask(mask: np.ndarray, scale: int, dilate: int = 0) -> np.ndarray:
    scaled = cv2.resize(
        mask,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    if dilate > 0:
        kernel = np.ones((dilate, dilate), dtype=np.uint8)
        scaled = cv2.dilate(scaled, kernel)
    return scaled


def dump_hp_ocr_debug(
    debug_dir: Path,
    *,
    frame: np.ndarray,
    config: dict[str, Any],
    label: str | None = None,
    area: dict[str, Any] | None = None,
    debug_images: list[tuple[str, np.ndarray]] | None = None,
    otsu_threshold: float | None = None,
    variants: list[tuple[str, np.ndarray]] | None = None,
    recognized: list[str] | None = None,
    result: tuple[int, int] | None = None,
    error: BaseException | None = None,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    log_path = debug_dir / "details.txt"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"time={datetime.now().isoformat(timespec='milliseconds')}\n")
        if label:
            log.write(f"source={label}\n")
        log.write(f"frame {_image_stats(frame)}\n")
        try:
            version = pytesseract.get_tesseract_version()
        except Exception as exc:
            version = f"недоступна ({exc})"
        log.write(f"tesseract_cmd={tesseract_cmd_from_config(config)!r}\n")
        log.write(f"tesseract_version={version}\n")
        log.write(f"ocr_config={OCR_CONFIG!r}\n")
        log.write(f"ocr_probe_config={OCR_CONFIG_NO_WHITELIST!r}\n")

        if area is None:
            log.write("enemy_hp_area: отсутствует в конфиге\n")
        else:
            left = int(area.get("left", -1))
            top = int(area.get("top", -1))
            right = int(area.get("right", -1))
            bottom = int(area.get("bottom", -1))
            height, width = frame.shape[:2]
            in_bounds = 0 <= left < right <= width and 0 <= top < bottom <= height
            log.write(
                f"enemy_hp_area left={left} top={top} right={right} "
                f"bottom={bottom} width={right - left} height={bottom - top} "
                f"in_bounds={in_bounds}\n"
            )
            try:
                _save_marked_frame(
                    debug_dir / "00_frame_marked.png",
                    frame,
                    left,
                    top,
                    right,
                    bottom,
                    in_bounds,
                )
                log.write("saved=00_frame_marked.png\n")
            except Exception:
                log.write("failed=00_frame_marked.png\n")
                log.write(traceback.format_exc())

        for name, image in debug_images or []:
            log.write(f"{name}: {_image_stats(image)}\n")
            try:
                _write_image(debug_dir / name, image)
                log.write(f"saved={name}\n")
            except Exception:
                log.write(f"failed={name}\n")
                log.write(traceback.format_exc())

        if otsu_threshold is not None:
            log.write(f"otsu_threshold={otsu_threshold}\n")

        if variants:
            for title, image in variants:
                try:
                    algo = _ocr_dump(image, OCR_CONFIG)
                    probe = _ocr_dump(image, OCR_CONFIG_NO_WHITELIST)
                except Exception:
                    log.write(f"\n=== {title} OCR ошибка ===\n")
                    log.write(traceback.format_exc())
                    continue
                _write_ocr_result(log, f"{title} (алгоритм)", algo)
                _write_ocr_result(log, f"{title} (без whitelist, только проба)", probe)

        if recognized is not None:
            log.write(f"\nalgorithm_raw_texts={recognized!r}\n")
        if result is not None:
            log.write(f"result={result[0]}/{result[1]}\n")
        else:
            log.write("result=None\n")
        if error is not None:
            log.write("\nexception:\n")
            log.write("".join(traceback.format_exception(error)))

    logger.debug("Отладка OCR HP сохранена в {}", debug_dir)


def read_enemy_hp(
    frame: np.ndarray,
    config: dict[str, Any],
    *,
    debug_dir: Path | str | None = None,
    debug_label: str | None = None,
) -> tuple[int, int] | None:
    """Читает HP мискрита в формате current/total."""
    dump_dir = Path(debug_dir) if debug_dir else None
    area: dict[str, Any] | None = None
    debug_images: list[tuple[str, np.ndarray]] = []
    otsu_threshold: float | None = None
    variants: list[tuple[str, np.ndarray]] = []
    recognized: list[str] = []
    result: tuple[int, int] | None = None
    error: BaseException | None = None

    try:
        area = config["enemy_hp_area"]
        left = int(area["left"])
        top = int(area["top"])
        right = int(area["right"])
        bottom = int(area["bottom"])
        ref = config.get("reference_resolution")
        ref = ref if isinstance(ref, dict) else {}
        layout = FrameLayout.from_frame(
            frame.shape[1],
            frame.shape[0],
            int(ref.get("width", 1920)),
            int(ref.get("height", 1080)),
            str(ref.get("fit", "fill")),
        )
        left, top, right, bottom = layout.ref_to_frame_rect(left, top, right, bottom)
        if not (
            0 <= left < right <= frame.shape[1]
            and 0 <= top < bottom <= frame.shape[0]
        ):
            raise ValueError(
                "область HP мискрита находится вне снимка "
                f"({left},{top})–({right},{bottom}), кадр "
                f"{frame.shape[1]}x{frame.shape[0]}"
            )

        tesseract_cmd = tesseract_cmd_from_config(config)
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        crop = frame[top:bottom, left:right]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(
            gray,
            None,
            fx=5,
            fy=5,
            interpolation=cv2.INTER_CUBIC,
        )
        otsu_threshold, binary = cv2.threshold(
            enlarged,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        inverted = cv2.bitwise_not(binary)
        white_mask = _white_text_mask(crop)
        white_mask_x5 = _scale_white_mask(white_mask, 5)
        white_mask_thick = _scale_white_mask(
            white_mask,
            WHITE_TEXT_SCALE,
            WHITE_TEXT_DILATE,
        )
        white_mask_thick_inv = cv2.bitwise_not(white_mask_thick)
        bright_200 = _bright_text_mask(crop, 200)
        bright_220 = _bright_text_mask(crop, 220)
        bright_200_x5 = _scale_white_mask(bright_200, 5)
        bright_220_x5 = _scale_white_mask(bright_220, 5)
        debug_images = [
            ("01_crop.png", crop),
            ("02_gray.png", gray),
            ("03_enlarged.png", enlarged),
            ("04_binary.png", binary),
            ("05_binary_inverted.png", inverted),
            ("06_white_mask.png", white_mask),
            ("07_white_mask_x5.png", white_mask_x5),
            ("08_white_mask_x8_dilate.png", white_mask_thick),
            ("09_white_mask_x8_dilate_inverted.png", white_mask_thick_inv),
            ("10_bright_200_x5.png", bright_200_x5),
            ("11_bright_220_x5.png", bright_220_x5),
        ]
        variants = [
            ("bright_200_x5", bright_200_x5),
            ("bright_220_x5", bright_220_x5),
            ("white_mask_x8_dilate", white_mask_thick),
            ("white_mask_x8_dilate_inverted", white_mask_thick_inv),
            ("white_mask_x5", white_mask_x5),
            ("binary", binary),
            ("binary_inverted", inverted),
            ("enlarged", enlarged),
        ]
        votes: list[tuple[int, int]] = []
        for name, image in variants:
            text = pytesseract.image_to_string(image, config=OCR_CONFIG).strip()
            recognized.append(text)
            reading = _parse_hp_reading(text)
            if reading is None:
                logger.debug("OCR HP {}: {!r} — нет допустимого current/total", name, text)
                continue
            votes.append(reading)
            logger.debug("OCR HP {}: {}/{} (текст={!r})", name, reading[0], reading[1], text)
        bar_full = _bar_looks_full(crop)
        result = _pick_hp_reading(votes, bar_full=bar_full)
        if result is not None:
            logger.info(
                "HP мискрита: {}/{} (голоса {}, полоска_полная={})",
                result[0],
                result[1],
                [f"{cur}/{total}" for cur, total in votes],
                bar_full,
            )
        else:
            logger.warning(
                "Не удалось прочитать HP мискрита в области "
                "({}, {})–({}, {}): {}",
                left,
                top,
                right,
                bottom,
                recognized,
            )
    except (
        KeyError,
        TypeError,
        ValueError,
        pytesseract.TesseractError,
        pytesseract.TesseractNotFoundError,
    ) as exc:
        error = exc
        logger.exception("Ошибка OCR HP мискрита")

    if dump_dir is not None:
        try:
            dump_hp_ocr_debug(
                dump_dir,
                frame=frame,
                config=config,
                label=debug_label,
                area=area,
                debug_images=debug_images,
                otsu_threshold=otsu_threshold,
                variants=variants,
                recognized=recognized,
                result=result,
                error=error,
            )
        except Exception:
            logger.exception("Не удалось сохранить отладку OCR HP в {}", dump_dir)

    return result
