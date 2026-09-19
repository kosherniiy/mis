from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock

import cv2
import numpy as np
from loguru import logger


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


def find_template(
    image: np.ndarray,
    template_path: str | Path,
    threshold: float = 0.85,
) -> tuple[int, int, float] | None:
    """Возвращает левый верхний угол лучшего совпадения и confidence."""
    template = load_template(template_path)
    if template is None:
        return None
    if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
        logger.error("Шаблон {} больше захваченного изображения", template_path)
        return None
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, location = cv2.minMaxLoc(result)
    if confidence < threshold:
        return None
    return location[0], location[1], float(confidence)


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
    ) -> None:
        self.templates_dir = Path(templates_dir)
        self.threshold = threshold
        self.debug = debug
        self.debug_dir = Path(debug_dir)
        if debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

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
            path = self.template_path(name)
            template = load_template(path)
            if template is None:
                return None
            found = find_template(image, path, threshold or self.threshold)
            if found is None:
                return None
            x, y, confidence = found
            match = TemplateMatch(x, y, template.shape[1], template.shape[0], confidence)
            logger.debug(
                "Шаблон {} найден: x={}, y={}, confidence={:.3f}",
                name,
                x,
                y,
                confidence,
            )
            if self.debug and save_debug:
                self.save_debug(image, state, name, match)
            return match
        except Exception:
            logger.exception("Ошибка поиска шаблона {}", name)
            return None

    def save_debug(
        self,
        image: np.ndarray,
        state: str,
        label: str | None = None,
        match: TemplateMatch | None = None,
    ) -> Path | None:
        if not self.debug:
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = self.debug_dir / f"{state.lower()}_{timestamp}.png"
            cv2.imwrite(str(path), output)
            logger.debug("Отладочный снимок сохранён: {}", path)
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
    ) -> Path | None:
        """Сохраняет снимок с хорошо заметной точкой выполненного клика."""
        if not self.debug:
            return None
        try:
            output = image.copy()
            cv2.circle(output, (x, y), 11, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(output, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.line(output, (x - 16, y), (x + 16, y), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(output, (x, y - 16), (x, y + 16), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(
                output,
                f"{label} ({x}, {y})",
                (max(4, x + 14), max(20, y - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = self.debug_dir / f"{state.lower()}_{timestamp}.png"
            cv2.imwrite(str(path), output)
            logger.info("Снимок после клика сохранён: {}", path)
            return path
        except Exception:
            logger.exception("Не удалось сохранить снимок с точкой клика")
            return None
