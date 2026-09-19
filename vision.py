from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock

import cv2
import numpy as np
from loguru import logger

from layout import FrameLayout


_TEMPLATE_CACHE: dict[Path, np.ndarray] = {}
_CACHE_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2


def load_template(template_path: str | Path) -> np.ndarray | None:
    """Загружает PNG и кэширует его между поисками."""
    path = Path(template_path).resolve()
    with _CACHE_LOCK:
        if path in _TEMPLATE_CACHE:
            return _TEMPLATE_CACHE[path]
        if not path.is_file():
            logger.error("Шаблон отсутствует: {}", path)
            return None
        template = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if template is None:
            logger.error("OpenCV не смог прочитать шаблон: {}", path)
            return None
        _TEMPLATE_CACHE[path] = template
        return template


def match_template_image(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.85,
) -> tuple[int, int, float] | None:
    """Возвращает левый верхний угол лучшего совпадения и confidence."""
    if (
        image.shape[0] < template.shape[0]
        or image.shape[1] < template.shape[1]
    ):
        return None
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, location = cv2.minMaxLoc(result)
    if confidence < threshold:
        return None
    return int(location[0]), int(location[1]), float(confidence)


def find_all_templates(
    image: np.ndarray,
    template_path: str | Path,
    threshold: float = 0.85,
    min_distance: int = 12,
) -> list[tuple[int, int, float]]:
    """Возвращает неперекрывающиеся совпадения, отсортированные по качеству."""
    template = load_template(template_path)
    if template is None:
        return []
    if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
        return []
    response = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(response >= threshold)
    candidates = sorted(
        ((int(x), int(y), float(response[y, x])) for x, y in zip(xs, ys)),
        key=lambda item: item[2],
        reverse=True,
    )
    selected: list[tuple[int, int, float]] = []
    for candidate in candidates:
        if all(
            abs(candidate[0] - existing[0]) > min_distance
            or abs(candidate[1] - existing[1]) > min_distance
            for existing in selected
        ):
            selected.append(candidate)
    return selected


class Vision:
    def __init__(
        self,
        templates_dir: str | Path,
        threshold: float = 0.85,
        debug: bool = False,
        debug_dir: str | Path = "./debug",
        reference_width: int = 1920,
        reference_height: int = 1080,
        layout_fit: str = "fill",
        layout_match: float = 1.0,
    ) -> None:
        self.templates_dir = Path(templates_dir)
        self.threshold = threshold
        self.debug = debug
        self.debug_dir = Path(debug_dir)
        self.reference_width = max(1, int(reference_width))
        self.reference_height = max(1, int(reference_height))
        self.layout_fit = str(layout_fit or "fill")
        self.layout_match = min(max(float(layout_match), 0.0), 1.0)
        self._layout_override: FrameLayout | None = None
        if debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def use_layout(self, layout: FrameLayout | None) -> None:
        self._layout_override = layout

    def _layout_for(self, image: np.ndarray) -> FrameLayout:
        override = self._layout_override
        if (
            override is not None
            and override.frame_w == image.shape[1]
            and override.frame_h == image.shape[0]
        ):
            return override
        return FrameLayout.from_frame(
            image.shape[1],
            image.shape[0],
            self.reference_width,
            self.reference_height,
            self.layout_fit,
            match=self.layout_match,
        )

    def _scale_template(self, template: np.ndarray, image: np.ndarray) -> np.ndarray:
        layout = self._layout_for(image)
        width, height = layout.ref_to_frame_size(template.shape[1], template.shape[0])
        if width == template.shape[1] and height == template.shape[0]:
            return template
        interpolation = (
            cv2.INTER_AREA
            if width < template.shape[1] or height < template.shape[0]
            else cv2.INTER_LINEAR
        )
        return cv2.resize(template, (width, height), interpolation=interpolation)

    def template_path(self, name: str) -> Path:
        return self.templates_dir / name

    def template_exists(self, name: str) -> bool:
        return self.template_path(name).is_file()

    def find(
        self,
        image: np.ndarray,
        name: str,
        threshold: float | None = None,
        state: str = "unknown",
        save_debug: bool = True,
    ) -> TemplateMatch | None:
        try:
            if self.debug:
                return self.probe_template(
                    image,
                    name,
                    threshold,
                    folder=f"templates/{Path(name).stem}",
                )
            path = self.template_path(name)
            template = load_template(path)
            if template is None:
                return None
            scaled = self._scale_template(template, image)
            found = match_template_image(image, scaled, threshold or self.threshold)
            if found is None:
                return None
            x, y, confidence = found
            match = TemplateMatch(x, y, scaled.shape[1], scaled.shape[0], confidence)
            logger.debug(
                "Шаблон {} найден: x={}, y={}, confidence={:.3f}",
                name,
                x,
                y,
                confidence,
            )
            if save_debug:
                self.save_debug(image, state, name, match)
            return match
        except Exception:
            logger.exception("Ошибка поиска шаблона {}", name)
            return None

    @staticmethod
    def _overlay_pixel_zoom(
        output: np.ndarray,
        image: np.ndarray,
        x: int,
        y: int,
        color: tuple[int, int, int],
        radius: int = 16,
        zoom: int = 8,
    ) -> None:
        height, width = image.shape[:2]
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return
        magnified = cv2.resize(
            crop,
            (crop.shape[1] * zoom, crop.shape[0] * zoom),
            interpolation=cv2.INTER_NEAREST,
        )
        local_x = (x - x0) * zoom + zoom // 2
        local_y = (y - y0) * zoom + zoom // 2
        cv2.circle(magnified, (local_x, local_y), max(3, zoom // 2), color, 1, cv2.LINE_AA)
        magnified = cv2.copyMakeBorder(
            magnified,
            2,
            2,
            2,
            2,
            cv2.BORDER_CONSTANT,
            value=color,
        )
        mh, mw = magnified.shape[:2]
        oh, ow = output.shape[:2]
        px = 8 if x >= width // 2 else max(8, ow - mw - 8)
        py = 8
        if px + mw > ow or py + mh > oh:
            return
        output[py : py + mh, px : px + mw] = magnified

    def save_debug(
        self,
        image: np.ndarray,
        state: str,
        label: str | None = None,
        match: TemplateMatch | None = None,
        force: bool = False,
    ) -> Path | None:
        if not self.debug and not force:
            return None
        try:
            output = image.copy()
            if match is not None:
                cv2.rectangle(
                    output,
                    (match.x, match.y),
                    (match.x + match.width, match.y + match.height),
                    (0, 255, 0),
                    2,
                )
                text = f"{label or 'match'} {match.confidence:.3f}"
                cv2.putText(
                    output,
                    text,
                    (match.x, max(18, match.y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = self.debug_dir / f"{state.lower()}_{timestamp}.png"
            cv2.imwrite(str(path), output)
            logger.info("Отладочный снимок сохранён: {}", path)
            return path
        except Exception:
            logger.exception("Не удалось сохранить отладочный снимок")
            return None

    def save_debug_point(
        self,
        image: np.ndarray,
        state: str,
        x: int,
        y: int,
        label: str = "click",
        hit: bool | None = None,
        keep: int = 80,
        force: bool = False,
    ) -> Path | None:
        """Снимок с точкой пикселя. При --debug пишется и HIT, и MISS."""
        if not self.debug and not force:
            return None
        try:
            output = image.copy()
            if hit is True:
                color = (0, 220, 0)
                status = "HIT"
            elif hit is False:
                color = (0, 0, 255)
                status = "MISS"
            else:
                color = (0, 0, 255)
                status = "POINT"
            cv2.circle(output, (x, y), 11, color, 2, cv2.LINE_AA)
            cv2.circle(output, (x, y), 3, color, -1, cv2.LINE_AA)
            cv2.line(output, (x - 16, y), (x + 16, y), color, 1, cv2.LINE_AA)
            cv2.line(output, (x, y - 16), (x, y + 16), color, 1, cv2.LINE_AA)
            header = f"{status} {label} ({x}, {y})"
            cv2.putText(
                output,
                header,
                (max(4, x + 14), max(20, y - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            self._overlay_pixel_zoom(output, image, x, y, color)
            out_dir = self.debug_dir / "pixels"
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_state = "".join(
                char if char.isalnum() or char in "-_" else "_" for char in state
            )
            path = out_dir / f"{status.lower()}_{safe_state}_{timestamp}.png"
            cv2.imwrite(str(path), output)
            logger.info("Снимок пикселя сохранён: {}", path)
            _prune_debug_dir(out_dir, keep)
            return path
        except Exception:
            logger.exception("Не удалось сохранить снимок с точкой клика")
            return None

    def save_debug_rect(
        self,
        image: np.ndarray,
        state: str,
        left: int,
        top: int,
        right: int,
        bottom: int,
        label: str = "",
        hit: bool | None = None,
        keep: int = 40,
        force: bool = False,
    ) -> Path | None:
        """Снимок с прямоугольником области поиска (OCR/зона)."""
        if not self.debug and not force:
            return None
        try:
            output = image.copy()
            if hit is True:
                color = (0, 220, 0)
                status = "HIT"
            elif hit is False:
                color = (0, 0, 255)
                status = "MISS"
            else:
                color = (0, 200, 255)
                status = "AREA"
            cv2.rectangle(output, (left, top), (right, bottom), color, 2)
            header = f"{status} {state} {label} ({left},{top})-({right},{bottom})"
            cv2.putText(
                output,
                header,
                (max(4, left), max(18, top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
            out_dir = self.debug_dir / "areas"
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_state = "".join(
                char if char.isalnum() or char in "-_" else "_" for char in state
            )
            path = out_dir / f"{status.lower()}_{safe_state}_{timestamp}.png"
            cv2.imwrite(str(path), output)
            logger.info("Снимок области сохранён: {}", path)
            _prune_debug_dir(out_dir, keep)
            return path
        except Exception:
            logger.exception("Не удалось сохранить снимок области")
            return None

    def probe_template(
        self,
        image: np.ndarray,
        name: str,
        threshold: float | None = None,
        folder: str = "template_probe",
        keep: int = 40,
    ) -> TemplateMatch | None:
        """Ищет шаблон, всегда сохраняет снимок лучшего совпадения — даже при промахе."""
        path = self.template_path(name)
        template = load_template(path)
        if template is None:
            return None
        scaled = self._scale_template(template, image)
        if image.shape[0] < scaled.shape[0] or image.shape[1] < scaled.shape[1]:
            logger.error(
                "Шаблон {} больше кадра {}x{}",
                name,
                image.shape[1],
                image.shape[0],
            )
            return None
        response = cv2.matchTemplate(image, scaled, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, location = cv2.minMaxLoc(response)
        x, y = int(location[0]), int(location[1])
        used = float(threshold or self.threshold)
        hit = float(confidence) >= used
        match = TemplateMatch(
            x,
            y,
            int(scaled.shape[1]),
            int(scaled.shape[0]),
            float(confidence),
        )
        saved = None
        if self.debug:
            saved = self._save_template_probe(
                image,
                scaled,
                match,
                name=name,
                threshold=used,
                hit=hit,
                folder=folder,
                keep=keep,
            )
        logger.info(
            "Проба {}: {} conf={:.3f}/{} at ({}, {}) кадр={}x{}{}",
            name,
            "HIT" if hit else "MISS",
            confidence,
            used,
            x,
            y,
            image.shape[1],
            image.shape[0],
            f" -> {saved}" if saved else "",
        )
        return match if hit else None

    def _save_template_probe(
        self,
        image: np.ndarray,
        template: np.ndarray,
        match: TemplateMatch,
        *,
        name: str,
        threshold: float,
        hit: bool,
        folder: str,
        keep: int,
    ) -> Path | None:
        try:
            frame_h, frame_w = image.shape[:2]
            tw, th = match.width, match.height
            crop = image[match.y : match.y + th, match.x : match.x + tw]
            annotated = image.copy()
            color = (0, 220, 0) if hit else (0, 0, 255)
            cv2.rectangle(
                annotated,
                (match.x, match.y),
                (match.x + tw, match.y + th),
                color,
                2,
            )
            bar_h = 92
            canvas = np.zeros((frame_h + bar_h, frame_w, 3), dtype=np.uint8)
            canvas[bar_h:] = annotated
            status = "HIT" if hit else "MISS"
            size_note = ""
            if frame_w != 1920 or frame_h != 1080:
                size_note = "  FRAME!=1920x1080"
            header = (
                f"{status}  {name}  conf={match.confidence:.3f}/{threshold:.2f}  "
                f"at ({match.x},{match.y})  frame={frame_w}x{frame_h}{size_note}"
            )
            cv2.putText(
                canvas,
                header,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "left=template  right=crop at best match",
                (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            thumb_h = 40
            template_thumb = _scale_to_height(template, thumb_h)
            crop_thumb = _scale_to_height(crop, thumb_h) if crop.size else template_thumb
            left = 8
            top = 52
            canvas[top : top + template_thumb.shape[0], left : left + template_thumb.shape[1]] = (
                template_thumb
            )
            gap = left + template_thumb.shape[1] + 10
            if gap + crop_thumb.shape[1] < frame_w:
                canvas[top : top + crop_thumb.shape[0], gap : gap + crop_thumb.shape[1]] = (
                    crop_thumb
                )
            out_dir = self.debug_dir / folder
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            out_path = out_dir / f"{status.lower()}_{timestamp}.png"
            cv2.imwrite(str(out_path), canvas)
            _prune_debug_dir(out_dir, keep)
            return out_path
        except Exception:
            logger.exception("Не удалось сохранить пробу шаблона {}", name)
            return None


def _scale_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.size == 0 or image.shape[0] == 0:
        return image
    scale = height / float(image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _prune_debug_dir(folder: Path, keep: int) -> None:
    files = sorted(folder.glob("*.png"), key=lambda item: item.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        try:
            old.unlink()
        except OSError:
            pass
