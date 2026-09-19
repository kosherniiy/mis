from __future__ import annotations

import random
import re
import sys
import threading
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from loguru import logger

from hp_ocr import read_enemy_hp
from notifier import NtfyNotifier
from runtime import GameWindowCapture, HumanInput, WaitKind, random_sleep
from session_manager import SessionManager
from tesseract_bin import resolve_tesseract_cmd
from vision import TemplateMatch, Vision


class BotState(Enum):
    IDLE = auto()
    MAP = auto()
    BATTLE = auto()
    RESULT = auto()
    LEVEL_UP = auto()
    LEVEL_UP_CONTINUE = auto()
    CHARACTER_PROMOTION = auto()
    TRAINING = auto()
    STOPPED_EPIC = auto()
    ERROR_PAUSE = auto()
    SHUTDOWN = auto()


class MiscritsBehavior:
    """State machine полного цикла фарма."""

    def __init__(self, config: dict[str, Any], stop_event: threading.Event) -> None:
        self.config = config
        self.stop_event = stop_event
        self.state = BotState.IDLE
        self.capture = GameWindowCapture(
            str(config.get("window_title", "Miscrits")),
            **self._capture_kwargs(config),
        )
        ref = config.get("reference_resolution")
        ref = ref if isinstance(ref, dict) else {}
        layout_fit = str(ref.get("fit", "fill"))
        if sys.platform == "darwin":
            macos = config.get("macos") if isinstance(config.get("macos"), dict) else {}
            layout_fit = str(macos.get("layout_fit") or "cover")
        self.vision = Vision(
            config.get("templates_dir", "./templates"),
            float(config.get("confidence_threshold", 0.85)),
            bool(config.get("debug", False)),
            "./debug",
            reference_width=int(ref.get("width", 1920)),
            reference_height=int(ref.get("height", 1080)),
            layout_fit=layout_fit,
            layout_match=(
                float(macos.get("layout_match", 0.5))
                if sys.platform == "darwin"
                else 1.0
            ),
            y_ref_height=(
                int(macos.get("ui_height", 1009))
                if sys.platform == "darwin"
                else 0
            ),
        )
        delays = config.get("action_delays", {})
        manual_pause = config.get("manual_mouse_pause", {})
        self.input = HumanInput(
            action_delay_min=float(delays.get("min", 0.8)),
            action_delay_max=float(delays.get("max", 2.5)),
            window_title=str(config.get("window_title", "Miscrits")),
            manual_pause_enabled=bool(manual_pause.get("enabled", True)),
            mouse_jerk_threshold_px=float(
                manual_pause.get("jerk_threshold_px", 80)
            ),
            mouse_poll_interval=float(manual_pause.get("poll_interval", 0.04)),
        )
        bind_window = getattr(self.input, "bind_window_checker", None)
        if callable(bind_window):
            bind_window(self.capture)
        self.session = SessionManager(config)
        self.notifier = NtfyNotifier(config.get("ntfy", {}))
        restored = self.session.state
        self.last_coordinates: tuple[int, int] | None = self._restore_coordinates(
            restored.get("last_coordinates")
        )
        self.template_misses = int(restored.get("template_misses", 0))
        self.map_object_misses = 0
        self.error_count = int(restored.get("error_count", 0))
        self.last_frame: np.ndarray | None = None
        self.pending_promotions = 0
        self.caught_this_battle = False

    @staticmethod
    def _capture_kwargs(config: dict[str, Any]) -> dict[str, Any]:
        ref = config.get("reference_resolution")
        ref = ref if isinstance(ref, dict) else {}
        kwargs: dict[str, Any] = {
            "reference_width": int(ref.get("width", 1920)),
            "reference_height": int(ref.get("height", 1080)),
            "layout_fit": str(ref.get("fit", "fill")),
        }
        if sys.platform != "darwin":
            return kwargs
        macos = config.get("macos") if isinstance(config.get("macos"), dict) else {}
        kwargs["titlebar_height"] = int(macos.get("titlebar_height", 0))
        kwargs["window_owner"] = str(macos.get("window_owner", ""))
        kwargs["layout_fit"] = str(macos.get("layout_fit") or "cover")
        kwargs["layout_match"] = float(macos.get("layout_match", 0.5))
        kwargs["y_ref_height"] = int(macos.get("ui_height", 1009))
        return kwargs

    @staticmethod
    def _restore_coordinates(value: object) -> tuple[int, int] | None:
        if (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(item, int) for item in value)
        ):
            return value[0], value[1]
        return None

    def transition(self, new_state: BotState, reason: str = "") -> None:
        if new_state is self.state:
            return
        logger.info(
            "Состояние: {} -> {}{}",
            self.state.name,
            new_state.name,
            f" ({reason})" if reason else "",
        )
        self.state = new_state

    def _ref_xy(self, x: int, y: int) -> tuple[int, int]:
        mapper = getattr(self.capture, "map_ref", None)
        if callable(mapper):
            return mapper(x, y)
        return x, y

    def _ref_rect(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> tuple[int, int, int, int]:
        mapper = getattr(self.capture, "map_ref_rect", None)
        if callable(mapper):
            return mapper(left, top, right, bottom)
        return left, top, right, bottom

    def run(self) -> None:
        logger.info("Запуск state machine Miscrits")
        start_phase = str(self.config.get("start_phase", "search"))
        start_states = {
            "search": BotState.MAP,
            "lvl_up_continue": BotState.LEVEL_UP_CONTINUE,
            "character_promotion": BotState.CHARACTER_PROMOTION,
        }
        initial_state = start_states.get(start_phase)
        if initial_state is None:
            logger.error("Неизвестная стартовая фаза: {}", start_phase)
            self.transition(BotState.SHUTDOWN, "ошибка стартовой фазы")
        else:
            self.transition(initial_state, f"стартовая фаза {start_phase}")
        handlers = {
            BotState.MAP: self._handle_map,
            BotState.BATTLE: self._handle_battle,
            BotState.RESULT: self._handle_result,
            BotState.LEVEL_UP: self._handle_level_up,
            BotState.LEVEL_UP_CONTINUE: self._handle_level_up_continue,
            BotState.CHARACTER_PROMOTION: self._handle_character_promotion,
            BotState.TRAINING: self._handle_training,
            BotState.STOPPED_EPIC: self._handle_stopped_epic,
            BotState.ERROR_PAUSE: self._handle_error_pause,
        }
        try:
            while (
                not self.stop_event.is_set()
                and not self.input.restart_requested.is_set()
                and self.state is not BotState.SHUTDOWN
            ):
                if self.state is not BotState.STOPPED_EPIC:
                    self.input.wait_if_paused()
                handler = handlers.get(self.state)
                if handler is None:
                    self.transition(BotState.SHUTDOWN, "нет обработчика состояния")
                    break
                try:
                    handler()
                except Exception:
                    logger.exception("Необработанная ошибка в состоянии {}", self.state.name)
                    self.transition(BotState.ERROR_PAUSE, "исключение обработчика")

            if not self.stop_event.is_set() and not self.input.restart_requested.is_set():
                logger.warning(
                    "Игровая логика остановлена, но бот остаётся активным "
                    "и ожидает команды перезапуска"
                )
                while (
                    not self.stop_event.is_set()
                    and not self.input.restart_requested.is_set()
                ):
                    self.stop_event.wait(0.25)
        finally:
            self.save_state()
            self.input.close()
            self.capture.close()

    def save_state(self) -> None:
        self.session.save_state(
            {
                "current_state": self.state.name,
                "last_coordinates": list(self.last_coordinates) if self.last_coordinates else None,
                "template_misses": self.template_misses,
                "error_count": self.error_count,
                "action_count": self.input.action_count,
            }
        )

    def _screen(self) -> np.ndarray | None:
        try:
            if self.input.restart_requested.is_set():
                self.stop_event.set()
                return None
            self.input.wait_if_paused()
            self.last_frame = self.capture.capture()
            binder = getattr(self.vision, "use_layout", None)
            if callable(binder):
                binder(getattr(self.capture, "layout", None))
            return self.last_frame
        except Exception:
            logger.exception("Ошибка захвата экрана")
            self.transition(BotState.ERROR_PAUSE, "ошибка захвата экрана")
            return None

    def _find(
        self,
        frame: np.ndarray,
        template: str,
        *,
        required: bool = False,
        save_debug: bool = True,
    ) -> TemplateMatch | None:
        match = self.vision.find(
            frame,
            template,
            state=self.state.name,
            save_debug=save_debug,
        )
        if match:
            if required:
                self.template_misses = 0
            return match
        if required:
            self._register_detection_miss(f"шаблон {template}")
        return None

    def _register_detection_miss(self, label: str) -> None:
        self.template_misses += 1
        maximum = int(self.config.get("max_template_misses", 5))
        logger.warning(
            "{} не определён ({}/{})",
            label,
            self.template_misses,
            maximum,
        )
        if self.template_misses >= maximum:
            if self.last_frame is not None:
                self.vision.save_debug(self.last_frame, "error", force=True)
            self.transition(BotState.ERROR_PAUSE, f"пропуски определения: {label}")

    def _click_match(self, match: TemplateMatch) -> tuple[int, int]:
        local_x, local_y = match.center
        screen_x, screen_y = self.capture.to_screen(local_x, local_y)
        actual_x, actual_y = self.input.click_human(screen_x, screen_y)
        self.last_coordinates = (actual_x, actual_y)
        if self.capture.last_region is None:
            return local_x, local_y
        return self.capture.from_screen(actual_x, actual_y)

    def _click_random_area(
        self,
        frame: np.ndarray,
        config_key: str,
        label: str,
        area_override: dict[str, Any] | None = None,
        apply_post_delay: bool = True,
        not_before: float | None = None,
        hold_range: tuple[float, float] | None = None,
        pre_click_delay_range: tuple[float, float] | None = None,
        not_before_reason: str = "интервал между атаками",
        not_before_wait_kind: WaitKind = "artificial",
        approach_if_farther_than: float | None = None,
        max_spread: float | None = None,
    ) -> bool:
        """Кликает в случайную безопасную точку области из конфига."""
        try:
            area = area_override if area_override is not None else self.config[config_key]
            left, top, right, bottom = self._ref_rect(
                int(area["left"]),
                int(area["top"]),
                int(area["right"]),
                int(area["bottom"]),
            )
            margin = max(0, int(area.get("margin", 8)))
        except (KeyError, TypeError, ValueError):
            logger.exception("Некорректный {} в config.yaml", config_key)
            self._register_detection_miss(label)
            return False

        safe_left, safe_right = left + margin, right - margin
        safe_top, safe_bottom = top + margin, bottom - margin
        if (
            safe_left >= safe_right
            or safe_top >= safe_bottom
            or left < 0
            or top < 0
            or right >= frame.shape[1]
            or bottom >= frame.shape[0]
        ):
            logger.error(
                "{} ({}, {})–({}, {}) вне снимка {}x{} или слишком мала",
                label,
                left,
                top,
                right,
                bottom,
                frame.shape[1],
                frame.shape[0],
            )
            self._register_detection_miss(label)
            return False

        # Треугольное распределение чаще попадает ближе к центру кнопки,
        # сохраняя естественный разброс по всей безопасной области.
        local_x = int(random.triangular(safe_left, safe_right, (safe_left + safe_right) / 2))
        local_y = int(random.triangular(safe_top, safe_bottom, (safe_top + safe_bottom) / 2))
        screen_x, screen_y = self.capture.to_screen(local_x, local_y)
        if approach_if_farther_than is not None:
            cursor_x, cursor_y = self.input.cursor_position()
            distance = ((screen_x - cursor_x) ** 2 + (screen_y - cursor_y) ** 2) ** 0.5
            if distance > approach_if_farther_than:
                approach_x = screen_x + random.randint(-10, 10)
                approach_y = screen_y - random.randint(28, 52)
                logger.debug(
                    "{}: длинный путь {:.0f}px, короткий заход в ({}, {})",
                    label,
                    distance,
                    approach_x,
                    approach_y,
                )
                self.input.move_mouse_human(approach_x, approach_y)
                random_sleep(0.06, 0.16, "короткий заход на кнопку")
        actual_x, actual_y = self.input.click_human(
            screen_x,
            screen_y,
            apply_post_delay=apply_post_delay,
            not_before=not_before,
            hold_range=hold_range,
            pre_click_delay_range=pre_click_delay_range,
            not_before_reason=not_before_reason,
            not_before_wait_kind=not_before_wait_kind,
            max_spread=max_spread,
        )
        self.last_coordinates = (actual_x, actual_y)
        logger.info(
            "{}: local=({}, {}), screen=({}, {})",
            label,
            local_x,
            local_y,
            actual_x,
            actual_y,
        )
        return True

    def _click_level_next(self, frame: np.ndarray, label: str) -> bool:
        """Повторяет записанный жест: почти прямой заход и два клика без сдвига."""
        if not self._click_random_area(
            frame,
            "level_next_area",
            label,
            apply_post_delay=False,
            hold_range=(0.18, 0.26),
            pre_click_delay_range=(0.22, 0.35),
            max_spread=28,
        ):
            return False
        random_sleep(0.38, 0.52, "пауза перед повторным кликом Далее")
        actual_x, actual_y = self.input.click_stationary_human()
        self.last_coordinates = (actual_x, actual_y)
        logger.info(
            "Повторный клик «Далее» без сдвига: screen=({}, {})",
            actual_x,
            actual_y,
        )
        return True

    def _members_needing_level(self, frame: np.ndarray) -> list[tuple[int, int, int]]:
        """Находит членов команды, чей маркер ближе к жёлтому цвету."""
        try:
            level_config = self.config["level_up"]
            marker_pixels = level_config["marker_pixels"]
            colors = level_config["marker_colors"]

            def parse_hex(value: object) -> np.ndarray:
                normalized = str(value).strip().lstrip("#")
                if len(normalized) != 6:
                    raise ValueError(f"некорректный цвет {value}")
                return np.array(
                    (
                        int(normalized[0:2], 16),
                        int(normalized[2:4], 16),
                        int(normalized[4:6], 16),
                    ),
                    dtype=np.float32,
                )

            yellow = parse_hex(colors["yellow"])
            blue = parse_hex(colors["blue"])
            gray = parse_hex(colors["gray"])
            result: list[tuple[int, int, int]] = []

            for index, marker in enumerate(marker_pixels, start=1):
                x, y = self._ref_xy(int(marker["x"]), int(marker["y"]))
                if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                    raise ValueError(f"маркер #{index} ({x}, {y}) вне снимка")
                pixel_blue, pixel_green, pixel_red = (
                    int(value) for value in frame[y, x]
                )
                actual = np.array(
                    (pixel_red, pixel_green, pixel_blue),
                    dtype=np.float32,
                )
                distances = {
                    "yellow": float(np.linalg.norm(actual - yellow)),
                    "blue": float(np.linalg.norm(actual - blue)),
                    "gray": float(np.linalg.norm(actual - gray)),
                }
                nearest = min(distances, key=distances.get)
                actual_hex = f"#{pixel_red:02x}{pixel_green:02x}{pixel_blue:02x}"
                logger.info(
                    "Маркер мискрита #{} ({}, {}): {}, ближайший цвет={}, расстояния={}",
                    index,
                    x,
                    y,
                    actual_hex,
                    nearest,
                    {name: round(value, 1) for name, value in distances.items()},
                )
                self.vision.save_debug_point(
                    frame,
                    f"level_marker_{index}",
                    x,
                    y,
                    f"#{index} {nearest} {actual_hex}",
                    hit=nearest == "yellow",
                )
                if nearest == "yellow":
                    result.append((index, x, y))
            return result
        except (KeyError, TypeError, ValueError):
            logger.exception("Некорректные маркеры или цвета в level_up")
            self._register_detection_miss("цветовые маркеры прокачки")
            return []

    def _battle_result_visible(self, frame: np.ndarray) -> bool:
        """Проверяет точный цвет пикселя, сигнализирующего о конце боя."""
        try:
            pixel = self.config["battle_result_pixel"]
            x, y = self._ref_xy(int(pixel["x"]), int(pixel["y"]))
            normalized = str(pixel["color"]).strip().lstrip("#")
            tolerance = float(pixel.get("tolerance", 0))
            if len(normalized) != 6:
                raise ValueError("цвет должен быть записан как #RRGGBB")
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                raise ValueError(f"точка ({x}, {y}) находится вне снимка")

            blue, green, red = (int(value) for value in frame[y, x])
            expected = np.array(
                (
                    int(normalized[0:2], 16),
                    int(normalized[2:4], 16),
                    int(normalized[4:6], 16),
                ),
                dtype=np.float32,
            )
            actual = np.array((red, green, blue), dtype=np.float32)
            distance = float(np.linalg.norm(actual - expected))
            if distance <= tolerance:
                actual_hex = f"#{red:02x}{green:02x}{blue:02x}"
                logger.info(
                    "Конец боя определён по пикселю ({}, {}): {}, отклонение={:.1f}",
                    x,
                    y,
                    actual_hex,
                    distance,
                )
                self.vision.save_debug_point(
                    frame,
                    "battle_result",
                    x,
                    y,
                    f"result {actual_hex} d={distance:.1f}",
                    hit=True,
                )
                return True
            actual_hex = f"#{red:02x}{green:02x}{blue:02x}"
            self.vision.save_debug_point(
                frame,
                "battle_result",
                x,
                y,
                f"result {actual_hex} d={distance:.1f}",
                hit=False,
            )
            return False
        except (KeyError, TypeError, ValueError):
            logger.exception("Некорректный battle_result_pixel в config.yaml")
            self._register_detection_miss("пиксель завершения боя")
            return False

    def _heal_needed(self, frame: np.ndarray) -> bool:
        """Определяет необходимость лечения по цвету заданного пикселя."""
        try:
            pixel = self.config["heal_check_pixel"]
            x, y = self._ref_xy(int(pixel["x"]), int(pixel["y"]))
            normalized = str(pixel["color"]).strip().lstrip("#")
            tolerance = float(pixel.get("tolerance", 20))
            if len(normalized) != 6:
                raise ValueError("цвет должен быть записан как #RRGGBB")
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                raise ValueError(f"точка ({x}, {y}) находится вне снимка")

            blue, green, red = (int(value) for value in frame[y, x])
            expected = np.array(
                (
                    int(normalized[0:2], 16),
                    int(normalized[2:4], 16),
                    int(normalized[4:6], 16),
                ),
                dtype=np.float32,
            )
            actual = np.array((red, green, blue), dtype=np.float32)
            distance = float(np.linalg.norm(actual - expected))
            actual_hex = f"#{red:02X}{green:02X}{blue:02X}"
            needed = distance <= tolerance
            logger.info(
                "Проверка лечения по пикселю ({}, {}): цвет={}, "
                "отклонение={:.1f}, требуется={}",
                x,
                y,
                actual_hex,
                distance,
                needed,
            )
            self.vision.save_debug_point(
                frame,
                "heal_check",
                x,
                y,
                f"{actual_hex} d={distance:.1f} need={needed}",
                hit=needed,
            )
            return needed
        except (KeyError, TypeError, ValueError):
            logger.exception("Некорректный heal_check_pixel в config.yaml")
            self._register_detection_miss("пиксель необходимости лечения")
            return False

    def _handle_map(self) -> None:
        if not self.capture.focus():
            self.transition(BotState.ERROR_PAUSE, "окно не сфокусировано")
            return
        random_sleep(0.3, 0.9, "после фокусировки окна")

        frame = self._screen()
        if frame is None:
            return
        target = self._find(frame, "map_wild_object.png")
        if target is None:
            self.map_object_misses += 1
            close_config = self.config.get("map_achievement_close", {})
            misses_before_click = max(
                1,
                int(close_config.get("misses_before_click", 5)),
            )
            logger.warning(
                "Объект карты не найден подряд: {}/{}",
                self.map_object_misses,
                misses_before_click,
            )
            if self.map_object_misses >= misses_before_click:
                local_x, local_y = self._ref_xy(
                    int(close_config.get("x", 942)),
                    int(close_config.get("y", 665)),
                )
                screen_x, screen_y = self.capture.to_screen(local_x, local_y)
                actual_x, actual_y = self.input.click_human(screen_x, screen_y)
                self.last_coordinates = (actual_x, actual_y)
                self.map_object_misses = 0
                logger.info(
                    "Закрытие предполагаемого окна достижения: "
                    "local=({}, {}), screen=({}, {})",
                    local_x,
                    local_y,
                    actual_x,
                    actual_y,
                )
            random_sleep(1, 3, "объект карты не найден")
            return

        self.map_object_misses = 0
        maximum_attempts = max(1, int(self.config.get("max_map_click_attempts", 20)))
        for click_attempt in range(1, maximum_attempts + 1):
            if self.stop_event.is_set():
                return
            logger.info(
                "Клик по объекту карты в фиксированной точке, попытка {}/{}",
                click_attempt,
                maximum_attempts,
            )
            if click_attempt == 1:
                click_x, click_y = self._click_match(target)
                post_click_frame = self._screen()
                if post_click_frame is None:
                    return
                self.vision.save_debug_point(
                    post_click_frame,
                    "map_click",
                    click_x,
                    click_y,
                    "фиксированная точка клика",
                )
            else:
                screen_x, screen_y = self.input.click_stationary_human()
                self.last_coordinates = (screen_x, screen_y)

            # Интервал отсчитывается от момента отпускания кнопки.
            interval = self.config.get("map_click_interval", {})
            check_after = random.uniform(
                float(interval.get("min", 4.0)),
                float(interval.get("max", 6.0)),
            )
            elapsed = time.monotonic() - self.input.last_click_at
            remaining = max(0.0, check_after - elapsed)
            if remaining > 0:
                random_sleep(
                    remaining,
                    remaining,
                    "ожидание battle_indicator после клика",
                )
            detected = self._wait_after_map_click()
            if detected:
                return
            logger.warning(
                "Через {:.2f} сек бой не начался после клика {}/{}",
                check_after,
                click_attempt,
                maximum_attempts,
            )

        self.transition(
            BotState.ERROR_PAUSE,
            f"{maximum_attempts} кликов в фиксированной точке без начала боя",
        )

    def _wait_after_map_click(self) -> bool:
        if self.stop_event.is_set():
            return True
        frame = self._screen()
        if frame is None:
            return True
        if self._find(frame, "battle_indicator.png"):
            self.transition(BotState.BATTLE, "обнаружен индикатор боя")
            return True
        return False

    def _detect_rarity(self, frame: np.ndarray) -> str | None:
        try:
            pixel_config = self.config.get("rarity_pixel", {})
            x, y = self._ref_xy(int(pixel_config["x"]), int(pixel_config["y"]))
            colors = pixel_config.get("colors", {})
            tolerance = float(pixel_config.get("tolerance", 24))
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                logger.error(
                    "Точка редкости ({}, {}) вне снимка {}x{}",
                    x,
                    y,
                    frame.shape[1],
                    frame.shape[0],
                )
                self.vision.save_debug(frame, "rarity_pixel_oob", force=True)
                return None

            blue, green, red = (int(value) for value in frame[y, x])
            actual = np.array((red, green, blue), dtype=np.float32)
            actual_hex = f"#{red:02x}{green:02x}{blue:02x}"
            candidates: list[tuple[str, float]] = []

            for rarity, color in colors.items():
                normalized = str(color).strip().lstrip("#")
                if len(normalized) != 6:
                    logger.error("Некорректный HEX для {}: {}", rarity, color)
                    continue
                expected = np.array(
                    (
                        int(normalized[0:2], 16),
                        int(normalized[2:4], 16),
                        int(normalized[4:6], 16),
                    ),
                    dtype=np.float32,
                )
                distance = float(np.linalg.norm(actual - expected))
                candidates.append((str(rarity), distance))

            if not candidates:
                logger.error("В rarity_pixel.colors нет корректных цветов")
                return None
            rarity, distance = min(candidates, key=lambda item: item[1])
            matched = distance <= tolerance
            self.vision.save_debug_point(
                frame,
                "rarity_pixel",
                x,
                y,
                f"{rarity} {actual_hex} d={distance:.1f}",
                hit=matched,
                force=not matched,
            )
            if matched:
                self.template_misses = 0
                logger.info(
                    "Редкость по пикселю ({}, {}): {}, цвет={}, отклонение={:.1f}",
                    x,
                    y,
                    rarity,
                    actual_hex,
                    distance,
                )
                return rarity

            logger.warning(
                "Неизвестный цвет редкости {} в ({}, {}); ближайший {} с отклонением {:.1f}, допуск {:.1f}",
                actual_hex,
                x,
                y,
                rarity,
                distance,
                tolerance,
            )
        except (KeyError, TypeError, ValueError):
            logger.exception("Некорректная настройка rarity_pixel в config.yaml")
        return None

    def _read_capture_chance(self, frame: np.ndarray) -> int | None:
        """Читает процент вероятности поимки из фиксированной области."""
        try:
            area = self.config["capture_chance_area"]
            left, top, right, bottom = self._ref_rect(
                int(area["left"]),
                int(area["top"]),
                int(area["right"]),
                int(area["bottom"]),
            )
            if not (
                0 <= left < right <= frame.shape[1]
                and 0 <= top < bottom <= frame.shape[0]
            ):
                raise ValueError("область вероятности поимки находится вне снимка")

            tesseract_cmd = resolve_tesseract_cmd(
                str(area.get("tesseract_cmd", "")).strip()
            )
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
            _, binary = cv2.threshold(
                enlarged,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            variants = (enlarged, binary, cv2.bitwise_not(binary))
            ocr_config = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789%"
            recognized: list[str] = []
            chance_found: int | None = None
            for image in variants:
                text = pytesseract.image_to_string(image, config=ocr_config).strip()
                recognized.append(text)
                match = re.search(r"\d{1,3}", text)
                if match is None:
                    continue
                chance = int(match.group())
                if 0 <= chance <= 100:
                    chance_found = chance
                    logger.info("Вероятность поимки мискрита: {}%", chance)
                    break

            self.vision.save_debug_rect(
                frame,
                "capture_chance",
                left,
                top,
                right,
                bottom,
                label=(
                    f"{chance_found}%"
                    if chance_found is not None
                    else f"ocr={recognized}"
                ),
                hit=chance_found is not None,
                force=chance_found is None,
            )
            if chance_found is not None:
                return chance_found

            logger.warning(
                "Не удалось прочитать вероятность поимки в области "
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
        ):
            logger.exception("Ошибка OCR вероятности поимки мискрита")
        return None

    def _read_enemy_hp(self, frame: np.ndarray) -> tuple[int, int] | None:
        hp = read_enemy_hp(frame, self.config)
        try:
            area = self.config.get("enemy_hp_area", {})
            left, top, right, bottom = self._ref_rect(
                int(area["left"]),
                int(area["top"]),
                int(area["right"]),
                int(area["bottom"]),
            )
            self.vision.save_debug_rect(
                frame,
                "enemy_hp",
                left,
                top,
                right,
                bottom,
                label="" if hp is None else f"{hp[0]}/{hp[1]}",
                hit=hp is not None,
            )
        except (KeyError, TypeError, ValueError):
            pass
        return hp

    def _rarity_below_epic(self, rarity: str) -> bool:
        order = [str(item) for item in self.config.get("rarity_order", [])]
        if "epic" not in order or rarity not in order:
            return rarity in {"common", "rare"}
        return order.index(rarity) < order.index("epic")

    def _pixel_bgr(self, frame: np.ndarray, x: int, y: int) -> tuple[int, int, int]:
        blue, green, red = (int(value) for value in frame[y, x])
        return blue, green, red

    def _catch_button_is_new(self, frame: np.ndarray) -> bool:
        """Анимация пикселя кнопки «Поймать» означает, что мискрита ещё нет в коллекции."""
        try:
            check = self.config.get("catch_button_new_check", {})
            x, y = self._ref_xy(int(check.get("x", 879)), int(check.get("y", 168)))
            duration = float(check.get("duration", 1.0))
            interval = float(check.get("interval", 0.2))
            if interval <= 0:
                raise ValueError("интервал проверки пикселя поимки должен быть > 0")
            sample_count = max(2, int(round(duration / interval)))
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                raise ValueError("пиксель проверки поимки вне снимка")

            samples: list[tuple[int, int, int]] = []
            current = frame
            for index in range(sample_count):
                if index > 0:
                    random_sleep(interval, interval, "проверка пикселя кнопки Поймать")
                    later_frame = self._screen()
                    if later_frame is None:
                        return False
                    current = later_frame
                    if not (
                        0 <= x < current.shape[1]
                        and 0 <= y < current.shape[0]
                    ):
                        return False
                samples.append(self._pixel_bgr(current, x, y))

            unique_colors = set(samples)
            all_same = len(unique_colors) == 1
            sample_hex = [
                f"#{red:02X}{green:02X}{blue:02X}"
                for blue, green, red in samples
            ]
            logger.info(
                "Пиксель поимки ({}, {}): {} ({} проб), все одинаковые={}, "
                "в коллекции={}",
                x,
                y,
                sample_hex,
                sample_count,
                all_same,
                all_same,
            )
            is_new = not all_same
            self.vision.save_debug_point(
                current,
                "catch_button_new",
                x,
                y,
                f"new={is_new} {sample_hex[-1] if sample_hex else ''}",
                hit=is_new,
            )
            return is_new
        except (KeyError, TypeError, ValueError):
            logger.exception("Не удалось проверить пиксель кнопки «Поймать»")
            return False

    def _players_turn_visible(self, frame: np.ndarray) -> bool:
        """Проверяет наличие надписи «Ваш ход!» в заданной области."""
        try:
            area = dict(self.config["turn_prompt_area"])
            if sys.platform == "darwin":
                macos = self.config.get("macos")
                override = macos.get("turn_prompt_area") if isinstance(macos, dict) else None
                if isinstance(override, dict):
                    area.update(override)
            left, top, right, bottom = self._ref_rect(
                int(area["left"]),
                int(area["top"]),
                int(area["right"]),
                int(area["bottom"]),
            )
            if not (
                0 <= left < right <= frame.shape[1]
                and 0 <= top < bottom <= frame.shape[0]
            ):
                raise ValueError("область надписи хода находится вне снимка")

            tesseract_cmd = resolve_tesseract_cmd(
                str(
                    self.config.get("capture_chance_area", {}).get(
                        "tesseract_cmd",
                        "",
                    )
                ).strip()
            )
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

            crop = frame[top:bottom, left:right]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            enlarged = cv2.resize(
                gray,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC,
            )
            _, binary = cv2.threshold(
                enlarged,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            for image in (enlarged, binary, cv2.bitwise_not(binary)):
                text = pytesseract.image_to_string(
                    image,
                    lang="rus",
                    config="--psm 7 --oem 3",
                )
                normalized = re.sub(r"[^а-я]", "", text.casefold().replace("ё", "е"))
                if "вашход" in normalized:
                    self.vision.save_debug_rect(
                        frame,
                        "turn_prompt",
                        left,
                        top,
                        right,
                        bottom,
                        label="ваш ход",
                        hit=True,
                    )
                    return True
            self.vision.save_debug_rect(
                frame,
                "turn_prompt",
                left,
                top,
                right,
                bottom,
                label="не найден",
                hit=False,
            )
            return False
        except (
            KeyError,
            TypeError,
            ValueError,
            pytesseract.TesseractError,
            pytesseract.TesseractNotFoundError,
        ):
            logger.exception("Ошибка OCR надписи «Ваш ход!»")
            return False

    def _handle_battle(self) -> None:
        frame = self._screen()
        if frame is None:
            return
        self.caught_this_battle = False
        rarity = self._detect_rarity(frame)
        if rarity is None:
            self._register_detection_miss("цвет редкости")
            if self.state is not BotState.ERROR_PAUSE:
                random_sleep(1, 3, "повтор определения редкости")
            return

        stop_rarity = {str(item) for item in self.config.get("stop_rarity", ["epic", "legendary"])}
        stop_capture_chances = self.config.get("stop_capture_chances", {})
        configured_chances = {
            int(value)
            for value in stop_capture_chances.get(rarity, [])
        }
        stop_for_rarity = (
            bool(self.config.get("stop_on_epic_plus", True))
            and rarity in stop_rarity
        )
        auto_catch_all = bool(self.config.get("autoCatchAll", False))
        auto_catch = bool(self.config.get("autoCatch", False))
        auto_catch_rares = bool(self.config.get("autoCatchRares", False))
        catch_all_rares = auto_catch_rares and rarity == "rare"
        pending_collection_check = self._rarity_below_epic(rarity) and (
            auto_catch or auto_catch_all or catch_all_rares
        )
        pending_encounter_scan = True
        should_auto_catch = auto_catch_all or catch_all_rares
        stall_any_rarity = bool(self.config.get("stallAnyRarity", False))
        is_new_to_collection = False
        capture_chance: int | None = None
        initial_capture_chance: int | None = None

        if should_auto_catch and not self.vision.template_exists("captured.png"):
            logger.error("Отсутствует обязательный шаблон captured.png")
            self._register_detection_miss("файл шаблона captured.png")
            self.transition(BotState.ERROR_PAUSE, "отсутствует captured.png")
            return
        if should_auto_catch:
            logger.info(
                "Автоловля включена: rarity={}, режим={}",
                rarity,
                "все" if auto_catch_all else "все rare",
            )

        attack_number = 0
        turn_check_interval = self.config.get("turn_check_interval", {})
        raw_sequence = self.config.get(
            "attack_sequence",
            [self.config.get("selected_attack", 2)],
        )
        if not isinstance(raw_sequence, (list, tuple)):
            raw_sequence = [raw_sequence]
        try:
            attack_sequence = [int(value) for value in raw_sequence]
            if not attack_sequence or any(
                attack not in (1, 2, 3, 4) for attack in attack_sequence
            ):
                raise ValueError("допустимы только атаки 1–4")
        except (TypeError, ValueError):
            logger.exception("Некорректная attack_sequence в config.yaml")
            self.transition(BotState.ERROR_PAUSE, "ошибка последовательности атак")
            return

        previous_attack: int | None = None
        catch_state = self._new_auto_catch_state()
        capture_kept = False
        while not self.stop_event.is_set() and self.state is BotState.BATTLE:
            frame = self._screen()
            if frame is None:
                return
            if self._battle_result_visible(frame):
                self._complete_victory_screen(attack_number)
                return
            if self.state is BotState.ERROR_PAUSE:
                return
            if should_auto_catch and not capture_kept and self._captured_visible(frame):
                if not self._handle_captured_window(
                    rarity,
                    initial_capture_chance,
                    is_new=is_new_to_collection,
                ):
                    return
                capture_kept = True
                continue

            if not self._players_turn_visible(frame):
                logger.info("Жду надпись «Ваш ход!», бой уже идёт")
                random_sleep(
                    float(turn_check_interval.get("min", 1.0)),
                    float(turn_check_interval.get("max", 1.5)),
                    "повторная проверка надписи Ваш ход",
                )
                continue

            if pending_collection_check:
                logger.info(
                    "Первый ход: проверка пикселя поимки, есть ли мискрит в коллекции"
                )
                missing_from_collection = self._catch_button_is_new(frame)
                pending_collection_check = False
                is_new_to_collection = missing_from_collection
                if missing_from_collection:
                    should_auto_catch = True
                    if not self.vision.template_exists("captured.png"):
                        logger.error("Отсутствует обязательный шаблон captured.png")
                        self._register_detection_miss("файл шаблона captured.png")
                        self.transition(BotState.ERROR_PAUSE, "отсутствует captured.png")
                        return
                    logger.info(
                        "Автоловля включена: rarity={}, режим=нет в коллекции",
                        rarity,
                    )
                else:
                    if should_auto_catch:
                        logger.info(
                            "Мискрит уже в коллекции, продолжаем ловлю: rarity={}",
                            rarity,
                        )
                    else:
                        logger.info(
                            "Мискрит уже в коллекции, обычный бой: rarity={}",
                            rarity,
                        )
                frame = self._screen()
                if frame is None:
                    return
                if self._battle_result_visible(frame):
                    self._complete_victory_screen(attack_number)
                    return
                if not self._players_turn_visible(frame):
                    continue

            if pending_encounter_scan:
                capture_chance = self._read_capture_chance(frame)
                initial_capture_chance = capture_chance
                shot = self._screen()
                if shot is not None:
                    frame = shot
                self._save_encounter_frame(
                    frame,
                    rarity,
                    capture_chance,
                )
                pending_encounter_scan = False
                stop_for_chance = (
                    capture_chance is not None
                    and capture_chance in configured_chances
                )
                if auto_catch and stop_for_chance:
                    should_auto_catch = True
                    if not self.vision.template_exists("captured.png"):
                        logger.error("Отсутствует обязательный шаблон captured.png")
                        self._register_detection_miss("файл шаблона captured.png")
                        self.transition(BotState.ERROR_PAUSE, "отсутствует captured.png")
                        return
                    logger.info(
                        "Автоловля включена: rarity={}, шанс={}, режим=особые шансы",
                        rarity,
                        capture_chance,
                    )
                if stall_any_rarity or (
                    (stop_for_rarity or stop_for_chance) and not should_auto_catch
                ):
                    chance_text = (
                        f", вероятность поимки {capture_chance}%"
                        if capture_chance is not None
                        else ""
                    )
                    if stall_any_rarity or stop_for_rarity:
                        wait_seconds = float(
                            self.config.get("epic_stall_wait_seconds", 120)
                        )
                        stall_attack = int(self.config.get("epic_stall_attack", 3))
                        if stall_any_rarity:
                            logger.warning(
                                "stallAnyRarity: удерживаем бой rarity={}",
                                rarity,
                            )
                        else:
                            self.notifier.send(
                                (
                                    f"Попался {rarity.upper()} мискрит{chance_text}. "
                                    f"Бот ждёт по {int(wait_seconds)} сек и жмёт "
                                    f"атаку {stall_attack} 1-го мувсета, пока не поставишь паузу."
                                ),
                                title="Особый мискрит",
                                priority=5,
                                tags=["warning", "video_game"],
                            )
                        self._stall_epic_encounter(rarity)
                        return
                    self.notifier.send(
                        (
                            f"Попался особый мискрит: {rarity.upper()}{chance_text}. "
                            "Бот остановил игровые действия."
                        ),
                        title="Особый мискрит",
                        priority=5,
                        tags=["warning", "video_game"],
                    )
                    self.transition(
                        BotState.STOPPED_EPIC,
                        f"{rarity} с вероятностью поимки {capture_chance}%",
                    )
                    return

            if should_auto_catch:
                ok, previous_attack, attack_number, capture_chance = (
                    self._handle_auto_catch_turn(
                        frame,
                        catch_state,
                        previous_attack,
                        attack_number,
                        capture_chance,
                        turn_check_interval,
                    )
                )
                if not ok:
                    return
                continue

            attack_number += 1
            selected_attack = attack_sequence[
                (attack_number - 1) % len(attack_sequence)
            ]
            logger.info("Атака №{}, ход {}", selected_attack, attack_number)
            if not self._click_battle_attack(frame, selected_attack, previous_attack):
                return
            previous_attack = selected_attack
            random_sleep(
                float(turn_check_interval.get("min", 1.0)),
                float(turn_check_interval.get("max", 1.5)),
                "проверка следующего хода после атаки",
            )

    def _new_auto_catch_state(self) -> dict[str, Any]:
        raw = self.config.get("auto_catch_moveset_attacks", [4, 4, 2])
        if not isinstance(raw, (list, tuple)) or not raw:
            attacks = [4, 4, 2]
        else:
            try:
                attacks = [int(value) for value in raw]
            except (TypeError, ValueError):
                attacks = [4, 4, 2]
        if any(attack not in (1, 2, 3, 4) for attack in attacks):
            logger.warning(
                "Некорректный auto_catch_moveset_attacks {}, используем [4, 4, 2]",
                raw,
            )
            attacks = [4, 4, 2]
        return {
            "moveset": 2 if bool(self.config.get("lowLevelEncounters", False)) else 1,
            "attacks": attacks,
            "last_damage": None,
            "current_hp": None,
            "hp_before_attack": None,
            "pending_moveset_switch": bool(
                self.config.get("lowLevelEncounters", False)
            ),
        }

    def _click_catch_button(self, frame: np.ndarray) -> bool:
        logger.info("Автоловля: нажатие кнопки «Ловить»")
        return self._click_random_area(
            frame,
            "catch_area",
            "Кнопка «Ловить»",
            apply_post_delay=False,
        )

    def _sleep_after_auto_catch_action(
        self,
        turn_check_interval: dict[str, Any],
        *,
        after_catch: bool,
    ) -> None:
        if after_catch:
            delay = self.config.get("capture_check_delay", {})
            random_sleep(
                float(delay.get("min", 0.8)),
                float(delay.get("max", 1.4)),
                "ожидание результата поимки",
            )
        random_sleep(
            float(turn_check_interval.get("min", 1.0)),
            float(turn_check_interval.get("max", 1.5)),
            "пауза после действия автоловли",
        )

    def _handle_auto_catch_turn(
        self,
        frame: np.ndarray,
        catch_state: dict[str, Any],
        previous_attack: int | None,
        attack_number: int,
        capture_chance: int | None,
        turn_check_interval: dict[str, Any],
    ) -> tuple[bool, int | None, int, int | None]:
        chance = self._read_capture_chance(frame)
        if chance is not None:
            capture_chance = chance
        if chance == 100:
            if not self._click_catch_button(frame):
                return False, previous_attack, attack_number, capture_chance
            self._sleep_after_auto_catch_action(turn_check_interval, after_catch=True)
            return True, None, attack_number, capture_chance

        hp_reading = self._read_enemy_hp(frame)
        if hp_reading is not None:
            current_hp, total_hp = hp_reading
            catch_state["current_hp"] = current_hp
            hp_before = catch_state.get("hp_before_attack")
            if hp_before is not None:
                catch_state["last_damage"] = hp_before - current_hp
                logger.info(
                    "Автоловля: HP {}/{} (было {}), урон {}, мувсет {}",
                    current_hp,
                    total_hp,
                    hp_before,
                    catch_state["last_damage"],
                    catch_state["moveset"],
                )
            else:
                logger.info(
                    "Автоловля: HP {}/{}, мувсет {}",
                    current_hp,
                    total_hp,
                    catch_state["moveset"],
                )

        current_hp = catch_state.get("current_hp")
        last_damage = catch_state.get("last_damage")
        attacks: list[int] = catch_state["attacks"]
        moveset = int(catch_state["moveset"])
        last_moveset = moveset >= len(attacks)

        if (
            isinstance(current_hp, int)
            and isinstance(last_damage, int)
            and last_damage > 0
        ):
            if last_moveset:
                threshold = 1.5 * last_damage
                if current_hp < threshold:
                    logger.info(
                        "Автоловля: HP {} < 1.5 × урон {} = {}, ловим",
                        current_hp,
                        last_damage,
                        threshold,
                    )
                    if not self._click_catch_button(frame):
                        return False, previous_attack, attack_number, capture_chance
                    self._sleep_after_auto_catch_action(
                        turn_check_interval,
                        after_catch=True,
                    )
                    return True, None, attack_number, capture_chance
            elif current_hp < 2 * last_damage:
                next_moveset = moveset + 1
                logger.info(
                    "Автоловля: HP {} < 2 × урон {} = {}, смена мувсета {} → {}",
                    current_hp,
                    last_damage,
                    2 * last_damage,
                    moveset,
                    next_moveset,
                )
                if not self._click_random_area(
                    frame,
                    "next_moveset_area",
                    "Следующий мувсет",
                    apply_post_delay=True,
                ):
                    return False, previous_attack, attack_number, capture_chance
                catch_state["moveset"] = next_moveset
                catch_state["last_damage"] = None
                previous_attack = None

        if catch_state.get("pending_moveset_switch"):
            logger.info("Автоловля: lowLevelEncounters, переход на 2-й мувсет")
            if not self._click_random_area(
                frame,
                "next_moveset_area",
                "Следующий мувсет",
                apply_post_delay=True,
            ):
                return False, previous_attack, attack_number, capture_chance
            catch_state["pending_moveset_switch"] = False
            previous_attack = None

        attacks = catch_state["attacks"]
        moveset = int(catch_state["moveset"])
        selected_attack = attacks[min(moveset, len(attacks)) - 1]
        attack_number += 1
        logger.info(
            "Автоловля: мувсет {}, атака №{}, ход {}",
            moveset,
            selected_attack,
            attack_number,
        )
        catch_state["hp_before_attack"] = catch_state.get("current_hp")
        if not self._click_battle_attack(frame, selected_attack, previous_attack):
            return False, previous_attack, attack_number, capture_chance
        self._sleep_after_auto_catch_action(turn_check_interval, after_catch=False)
        return True, selected_attack, attack_number, capture_chance

    def _click_battle_attack(
        self,
        frame: np.ndarray,
        selected_attack: int,
        previous_attack: int | None,
    ) -> bool:
        if selected_attack != previous_attack:
            return self._click_random_area(
                frame,
                f"attack_area_{selected_attack}",
                f"Атака №{selected_attack}",
                apply_post_delay=False,
            )
        actual_x, actual_y = self.input.click_stationary_human()
        self.last_coordinates = (actual_x, actual_y)
        return True

    def _captured_visible(self, frame: np.ndarray) -> bool:
        match = self._find(frame, "captured.png", save_debug=False)
        if match is None:
            return False
        self.vision.save_debug(frame, "captured", "captured.png", match)
        return True

    def _format_caught_miscrit_message(
        self,
        rarity: str,
        capture_chance: int | None,
        is_new: bool,
    ) -> str:
        parts = [f"Пойман {rarity.upper()} мискрит"]
        if self._rarity_below_epic(rarity):
            grades = {
                "common": {30: "A+", 27: "S+"},
                "rare": {20: "A+", 17: "S+"},
            }
            grade = grades.get(rarity.lower(), {}).get(capture_chance)
            if grade:
                parts.append(grade)
            if is_new:
                parts.append("Новый")
        return " ".join(parts)

    def _handle_captured_window(
        self,
        rarity: str,
        capture_chance: int | None,
        *,
        is_new: bool = False,
    ) -> bool:
        message = self._format_caught_miscrit_message(
            rarity,
            capture_chance,
            is_new,
        )
        logger.info(message)
        self.notifier.send(
            f"{message}.",
            title="Мискрит пойман",
            priority=4,
            tags=["tada", "video_game"],
        )
        frame = self._screen()
        if frame is None:
            return False
        if not self._click_random_area(
            frame,
            "keep_area",
            "Кнопка «Оставить»",
            apply_post_delay=False,
        ):
            return False
        delay = self.config.get("capture_check_delay", {})
        random_sleep(
            float(delay.get("min", 0.8)),
            float(delay.get("max", 1.4)),
            "закрытие окна поимки",
        )
        self.caught_this_battle = True
        return True

    def _complete_victory_screen(self, attack_number: int) -> None:
        if not self.vision.template_exists("lvl_up.png"):
            logger.error("Отсутствует обязательный шаблон lvl_up.png")
            self._register_detection_miss("файл шаблона lvl_up.png")
            self.transition(BotState.ERROR_PAUSE, "отсутствует lvl_up.png")
            return
        check_delay = self.config.get("level_up_result_check_delay", {})
        random_sleep(
            float(check_delay.get("min", 0.7)),
            float(check_delay.get("max", 1.2)),
            "ожидание отображения lvl_up после боя",
        )
        level_frame = self._screen()
        if level_frame is None:
            return
        self.pending_level_up = bool(self._find(level_frame, "lvl_up.png"))
        self.pending_heal = self._heal_needed(level_frame)
        logger.info(
            "Проверка lvl_up.png завершена до клика «Далее»: {}",
            "требуется прокачка" if self.pending_level_up else "прокачка не требуется",
        )
        logger.info("Нажатие кнопки «Далее» после {} атак", attack_number)
        if not self._click_random_area(
            level_frame,
            "continue_area",
            "Кнопка «Далее»",
        ):
            return
        self.transition(BotState.RESULT, "нажата кнопка «Далее»")

    def _close_visible_window(self, templates: tuple[str, ...]) -> bool:
        frame = self._screen()
        if frame is None:
            return False
        for template in templates:
            close = self._find(frame, template, save_debug=False)
            if close:
                self._click_match(close)
                return True
        return False

    def _handle_result(self) -> None:
        if not self._handle_optional_achievement():
            return
        if not self._handle_optional_character_up():
            return

        if not self.pending_level_up:
            if self.pending_heal and not self._handle_heal():
                return
            self.pending_heal = False
            self.transition(BotState.MAP, "lvl_up.png не обнаружен")
            return

        self.pending_heal = False
        self.transition(BotState.LEVEL_UP, "lvl_up.png был обнаружен")

    def _handle_heal(self) -> bool:
        """Открывает лечение команды и подтверждает действие."""
        frame = self._screen()
        if frame is None:
            return False
        logger.info("Требуется лечение: открытие окна подтверждения")
        if not self._click_random_area(
            frame,
            "heal_area",
            "Кнопка лечения",
            apply_post_delay=False,
        ):
            return False

        delay = self.config.get("heal_confirm_open_delay", {})
        random_sleep(
            float(delay.get("min", 0.5)),
            float(delay.get("max", 1.0)),
            "открытие подтверждения лечения",
        )
        frame = self._screen()
        if frame is None:
            return False
        if not self._click_random_area(
            frame,
            "heal_confirm_area",
            "Подтверждение лечения",
        ):
            return False
        logger.info("Лечение подтверждено")
        return True

    def _handle_optional_achievement(self) -> bool:
        """Закрывает необязательное сообщение о полученном достижении."""
        template = "achievment_ok.png"
        if not self.vision.template_exists(template):
            logger.warning("Шаблон {} отсутствует, проверка достижения пропущена", template)
            return True

        frame = self._screen()
        if frame is None:
            return False
        achievement = self._find(frame, template, save_debug=False)
        if achievement is None:
            logger.debug("Сообщение о достижении не появилось")
            return True

        self.vision.save_debug(frame, "achievement", template, achievement)
        logger.info("Обнаружено сообщение о получении достижения")
        return self._click_random_area(
            frame,
            "achievement_ok_area",
            "Подтверждение достижения",
        )

    def _handle_optional_character_up(self) -> bool:
        """Закрывает окно повышения персонажа после поимки, если оно появилось."""
        if not self.caught_this_battle:
            return True
        template = "character_up.png"
        if not self.vision.template_exists(template):
            logger.warning(
                "Шаблон {} отсутствует, проверка повышения персонажа пропущена",
                template,
            )
            return True

        self._level_sleep(
            "promotion_open",
            0.5,
            1.0,
            "появление окна повышения персонажа после поимки",
        )
        frame = self._screen()
        if frame is None:
            return False
        match = self._find(frame, template, save_debug=False)
        if match is None:
            logger.debug("Окно повышения персонажа после поимки не появилось")
            return True

        self.vision.save_debug(frame, "character_up", template, match)
        logger.info("Обнаружено окно повышения персонажа после поимки")
        if not self._click_character_promotion():
            return False
        self.caught_this_battle = False
        return True

    def _level_sleep(
        self,
        key: str,
        default_min: float,
        default_max: float,
        reason: str,
    ) -> None:
        delay = self.config.get("level_up", {}).get("delays", {}).get(key, {})
        random_sleep(
            float(delay.get("min", default_min)),
            float(delay.get("max", default_max)),
            reason,
        )

    def _handle_optional_enchant(self) -> bool:
        """Закрывает необязательное окно прокачки способности, если оно появилось."""
        if not self.vision.template_exists("enchant.png"):
            logger.error("Отсутствует обязательный шаблон enchant.png")
            self._register_detection_miss("файл шаблона enchant.png")
            self.transition(BotState.ERROR_PAUSE, "отсутствует enchant.png")
            return False

        frame = self._screen()
        if frame is None:
            return False
        enchant = self._find(frame, "enchant.png", save_debug=False)
        if enchant is not None:
            self.vision.save_debug(
                frame,
                "enchant",
                "enchant.png",
                enchant,
            )
            logger.info("Обнаружено окно прокачки способности")
            if not self._click_random_area(
                frame,
                "enchant_area",
                "Подтверждение прокачки способности",
                apply_post_delay=False,
            ):
                return False
            self._level_sleep(
                "popup_close",
                0.35,
                0.7,
                "закрытие окна прокачки способности",
            )
            return True
        logger.debug("Окно прокачки способности не появилось")
        return True

    def _click_character_promotion(self) -> bool:
        """Нажимает «Повышение» после завершившейся эволюции."""
        frame = self._screen()
        if frame is None:
            return False
        logger.info("Нажатие кнопки «Повышение» после эволюции")
        if not self._click_random_area(
            frame,
            "character_promotion_area",
            "Кнопка «Повышение»",
            apply_post_delay=False,
        ):
            return False
        self._level_sleep(
            "popup_close",
            0.35,
            0.7,
            "завершение повышения после эволюции",
        )
        return True

    def _finish_after_experience_close(self) -> bool:
        """Завершает прокачку после уже закрытого окна «Опыт»."""
        self._level_sleep(
            "achievement_after_experience",
            0.35,
            0.7,
            "появление достижения после закрытия окна опыта",
        )
        if not self._handle_optional_achievement():
            return False

        for index in range(self.pending_promotions):
            self._level_sleep(
                "promotion_open",
                0.5,
                1.0,
                f"открытие окна Повышение {index + 1}/{self.pending_promotions}",
            )
            logger.info(
                "Закрытие окна «Повышение» {}/{}",
                index + 1,
                self.pending_promotions,
            )
            if not self._click_character_promotion():
                return False
        self.pending_promotions = 0
        if not self._handle_optional_character_up():
            return False

        self._level_sleep("return_map", 0.5, 1.0, "возврат на карту")
        self.transition(BotState.MAP)
        return True

    def _handle_optional_evolve(self) -> bool:
        """Закрывает необязательное окно эволюции, если оно появилось."""
        if not self.vision.template_exists("evolve.png"):
            logger.error("Отсутствует обязательный шаблон evolve.png")
            self._register_detection_miss("файл шаблона evolve.png")
            self.transition(BotState.ERROR_PAUSE, "отсутствует evolve.png")
            return False

        frame = self._screen()
        if frame is None:
            return False
        evolve = self._find(frame, "evolve.png", save_debug=False)
        if evolve is not None:
            self.vision.save_debug(
                frame,
                "evolve",
                "evolve.png",
                evolve,
            )
            try:
                point = self.config["evolve_click"]
                local_x, local_y = self._ref_xy(int(point["x"]), int(point["y"]))
                if not (
                    0 <= local_x < frame.shape[1]
                    and 0 <= local_y < frame.shape[0]
                ):
                    raise ValueError("точка закрытия эволюции вне снимка")
            except (KeyError, TypeError, ValueError):
                logger.exception("Некорректный evolve_click в config.yaml")
                self._register_detection_miss("точка закрытия эволюции")
                return False

            screen_x, screen_y = self.capture.to_screen(local_x, local_y)
            actual_x, actual_y = self.input.click_human(
                screen_x,
                screen_y,
                apply_post_delay=False,
            )
            self.last_coordinates = (actual_x, actual_y)
            logger.info(
                "Окно эволюции закрыто: local=({}, {}), screen=({}, {})",
                local_x,
                local_y,
                actual_x,
                actual_y,
            )
            self._level_sleep("popup_close", 0.35, 0.7, "закрытие окна эволюции")
            self.pending_promotions += 1
            logger.info(
                "Эволюция закрыта, ожидается окон «Повышение»: {}",
                self.pending_promotions,
            )
            return True
        logger.debug("Окно эволюции не появилось")
        return True

    def _handle_character_promotion(self) -> None:
        """Стартует с уже открытого окна «Повышение»."""
        logger.info("Фаза character_promotion: открыто окно «Повышение»")
        if not self._click_character_promotion():
            return
        self.pending_promotions = 0
        self._level_sleep("return_map", 0.5, 1.0, "возврат на карту")
        self.transition(BotState.MAP)

    def _handle_level_up_continue(self) -> None:
        """Продолжает сценарий с уже открытого окна и кнопки «Далее»."""
        frame = self._screen()
        if frame is None:
            return

        logger.info("Фаза lvl_up_continue: нажатие уже открытой кнопки «Далее»")
        if not self._click_level_next(frame, "«Далее» стартовой фазы"):
            return
        self._level_sleep(
            "level_complete",
            0.35,
            0.75,
            "завершение прокачки мискрита",
        )
        if not self._handle_optional_evolve():
            return
        if not self._handle_optional_enchant():
            return
        if not self._handle_optional_evolve():
            return

        frame = self._screen()
        if frame is None:
            return
        logger.info("Фаза lvl_up_continue: закрытие окна опыта")
        if not self._click_random_area(
            frame,
            "training_close_area",
            "Закрытие окна опыта",
            apply_post_delay=False,
        ):
            return
        self._finish_after_experience_close()

    def _handle_level_up(self) -> None:
        self.pending_level_up = False
        self.pending_promotions = 0
        level_config = self.config.get("level_up", {})
        if not bool(level_config.get("enabled", True)):
            self.transition(BotState.MAP, "повышение уровня отключено")
            return

        frame = self._screen()
        if frame is None:
            return
        logger.info("Открытие вкладки «Опыт»")
        if not self._click_random_area(
            frame,
            "experience_area",
            "Кнопка «Опыт»",
            apply_post_delay=False,
        ):
            return

        self._level_sleep("experience_open", 0.6, 1.2, "открытие окна опыта")
        frame = self._screen()
        if frame is None:
            return
        members = self._members_needing_level(frame)
        if self.state is BotState.ERROR_PAUSE:
            return
        logger.info(
            "Прокачка требуется мискритам: {}",
            [member[0] for member in members] if members else "никому",
        )

        radius = level_config.get("marker_click_radius", {})
        radius_x = max(3, int(radius.get("x", 24)))
        radius_y = max(3, int(radius.get("y", 18)))
        for member_index, marker_x, marker_y in members:
            if self.stop_event.is_set():
                return
            frame = self._screen()
            if frame is None:
                return
            marker_area = {
                "left": marker_x - radius_x,
                "top": marker_y - radius_y,
                "right": marker_x + radius_x,
                "bottom": marker_y + radius_y,
                "margin": 2,
            }
            logger.info("Выбор мискрита #{} для прокачки", member_index)
            if not self._click_random_area(
                frame,
                "level_marker_area",
                f"Маркер мискрита #{member_index}",
                marker_area,
                apply_post_delay=False,
            ):
                return
            self._level_sleep("marker_select", 0.25, 0.6, "выбор мискрита")

            frame = self._screen()
            if frame is None:
                return
            if not self._click_random_area(
                frame,
                "level_action_area",
                f"Прокачать мискрита #{member_index}",
                apply_post_delay=False,
            ):
                return
            self._level_sleep(
                "level_window_open",
                0.7,
                1.3,
                "открытие окна прокачки",
            )

            frame = self._screen()
            if frame is None:
                return
            logger.info("Нажатие «Далее» для мискрита #{}", member_index)
            if not self._click_level_next(
                frame,
                f"«Далее» для мискрита #{member_index}",
            ):
                return
            self._level_sleep(
                "level_complete",
                0.35,
                0.75,
                "завершение прокачки мискрита",
            )
            if not self._handle_optional_evolve():
                return
            if not self._handle_optional_enchant():
                return
            if not self._handle_optional_evolve():
                return

        frame = self._screen()
        if frame is None:
            return
        logger.info("Закрытие окна опыта после прокачки команды")
        if not self._click_random_area(
            frame,
            "training_close_area",
            "Закрытие окна опыта",
            apply_post_delay=False,
        ):
            return
        self._finish_after_experience_close()

    def _handle_training(self) -> None:
        maximum = int(self.config.get("level_up", {}).get("max_training_attempts", 10))
        for attempt in range(1, maximum + 1):
            frame = self._screen()
            if frame is None:
                return
            training = self._find(frame, "training_button.png", save_debug=False)
            if training is None:
                logger.info("Тренировка больше недоступна после {} попыток", attempt - 1)
                break
            logger.info("Платиновая тренировка {}/{}", attempt, maximum)
            self._click_match(training)

            frame = self._screen()
            if frame is None:
                return
            platinum = self._find(frame, "training_platinum.png", required=True)
            if platinum is None:
                return
            self._click_match(platinum)

            frame = self._screen()
            if frame is None:
                return
            confirm = self._find(frame, "confirm_button.png", required=True)
            if confirm is None:
                return
            self._click_match(confirm)
            random_sleep(3, 7, "выполнение платиновой тренировки")

        self._close_all_windows()
        self.transition(BotState.MAP)

    def _close_all_windows(self) -> None:
        for _ in range(5):
            if self.stop_event.is_set():
                return
            if not self._close_visible_window(("close_info.png", "close_window.png")):
                random_sleep(0.3, 0.8, "проверка закрытия окон")
                frame = self._screen()
                if frame is None:
                    return
                if self._find(frame, "map_wild_object.png", save_debug=False):
                    return

    def _save_encounter_frame(
        self,
        frame: np.ndarray,
        rarity: str,
        capture_chance: int | None,
    ) -> Path | None:
        if not bool(self.config.get("debugEncounters", False)):
            return None
        folder = Path("./encounters")
        folder.mkdir(parents=True, exist_ok=True)
        chance = f"{capture_chance}p" if capture_chance is not None else "unknown"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        target = folder / f"{stamp}_{rarity}_{chance}.png"
        if cv2.imwrite(str(target), frame):
            logger.info("Скрин боя сохранён: {}", target)
            return target
        logger.warning("Не удалось сохранить скрин боя: {}", target)
        return None

    def _save_special_frame(self, prefix: str) -> Path | None:
        if not bool(self.config.get("debug", False)):
            return None
        frame = self.last_frame if self.last_frame is not None else self._screen()
        if frame is None:
            return None
        path = Path("./debug")
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png"
        if cv2.imwrite(str(target), frame):
            logger.info("Снимок сохранён: {}", target)
            return target
        return None

    def _wait_interruptible(self, seconds: float, reason: str) -> bool:
        """Ждёт seconds. Пауза игрока только приостанавливает таймер, бот не стопается."""
        logger.info("Ожидание {:.0f} сек: {}", seconds, reason)
        left = max(0.0, seconds)
        while left > 0:
            if self.stop_event.is_set() or self.input.restart_requested.is_set():
                return False
            self.input.wait_if_paused()
            if self.stop_event.is_set() or self.input.restart_requested.is_set():
                return False
            slice_s = min(0.25, left)
            started = time.monotonic()
            self.stop_event.wait(slice_s)
            left -= time.monotonic() - started
        return not (
            self.stop_event.is_set() or self.input.restart_requested.is_set()
        )

    def _stall_epic_encounter(self, rarity: str) -> None:
        """Держит бой: каждые 2 минуты атака 3 первого мувсета, пока бой не закончится."""
        wait_seconds = float(self.config.get("epic_stall_wait_seconds", 120))
        stall_attack = int(self.config.get("epic_stall_attack", 3))
        logger.warning(
            "Удержание боя rarity={}: каждые {:.0f} сек атака {} мувсета 1",
            rarity,
            wait_seconds,
            stall_attack,
        )
        previous_attack: int | None = None
        while (
            not self.stop_event.is_set()
            and not self.input.restart_requested.is_set()
            and self.state is BotState.BATTLE
        ):
            if not self._wait_interruptible(
                wait_seconds,
                f"ожидание игрока после {rarity}",
            ):
                return

            frame = self._screen()
            if frame is None:
                return
            if self._battle_result_visible(frame):
                self._complete_victory_screen(0)
                return
            if self._find(frame, "battle_indicator.png", save_debug=False) is None:
                self.transition(BotState.MAP, "бой с особым мискритом уже закрыт")
                return
            logger.info(
                "Удержание боя с {}: атака {} мувсета 1",
                rarity,
                stall_attack,
            )
            if not self._click_battle_attack(frame, stall_attack, previous_attack):
                return
            previous_attack = stall_attack

    def _handle_stopped_epic(self) -> None:
        self._save_special_frame("epic_stop")
        logger.critical("Найден мискрит, подходящий под условия остановки. Бот остановлен")
        while (
            not self.stop_event.is_set()
            and not self.input.restart_requested.is_set()
        ):
            self.stop_event.wait(0.25)
        self.transition(BotState.SHUTDOWN, "остановка пользователем")

    def _handle_error_pause(self) -> None:
        self.error_count += 1
        self._save_special_frame("error")
        maximum = int(self.config.get("max_errors_before_stop", 3))
        if self.error_count > maximum:
            logger.error("Превышен предел ошибок: {} > {}", self.error_count, maximum)
            self.notifier.send(
                (
                    f"Бот остановил игровые действия из-за ошибок: "
                    f"{self.error_count} > {maximum}."
                ),
                title="Ошибка бота Miscrits",
                priority=5,
                tags=["rotating_light", "warning"],
            )
            self.error_count = 0
            self.transition(
                BotState.SHUTDOWN,
                "ожидание ручного перезапуска после превышения предела ошибок",
            )
            return
        logger.warning("Пауза после ошибки, номер {}/{}", self.error_count, maximum)
        random_sleep(60, 180, "ERROR_PAUSE")
        self.template_misses = 0
        self.transition(BotState.MAP, "повтор после ошибки")
