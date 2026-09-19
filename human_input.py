from __future__ import annotations

import _thread
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

import pydirectinput
import win32api
import win32con
import win32gui
from loguru import logger


pydirectinput.PAUSE = 0


WaitKind = Literal["artificial", "forced"]


def random_sleep(
    min_sec: float,
    max_sec: float,
    reason: str | None = None,
    wait_kind: WaitKind | None = None,
) -> float:
    """Пауза случайной длины; все ожидания бота проходят через эту функцию."""
    low, high = sorted((max(0.0, min_sec), max(0.0, max_sec)))
    duration = random.uniform(low, high)
    started_at = time.monotonic()
    time.sleep(duration)
    actual_duration = time.monotonic() - started_at
    return actual_duration


@dataclass(slots=True)
class HumanInput:
    action_delay_min: float = 0.8
    action_delay_max: float = 2.5
    miss_chance: float = 0.05
    window_title: str = "Miscrits"
    manual_pause_enabled: bool = True
    mouse_jerk_threshold_px: float = 80.0
    mouse_poll_interval: float = 0.04
    action_count: int = field(init=False, default=0)
    last_click_at: float = field(init=False, default=0.0)
    _pause_event: threading.Event = field(init=False, default_factory=threading.Event)
    _shutdown_event: threading.Event = field(init=False, default_factory=threading.Event)
    restart_requested: threading.Event = field(init=False, default_factory=threading.Event)
    _resume_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _resume_required: bool = field(init=False, default=False)
    _pause_position: tuple[int, int] | None = field(init=False, default=None)
    _pause_source: str | None = field(init=False, default=None)
    _automation_mouse_event: threading.Event = field(
        init=False,
        default_factory=threading.Event,
    )
    _hotkey_thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        self.action_count = 0
        self._hotkey_thread = threading.Thread(
            target=self._monitor_pause_hotkey,
            name="bot-control-monitor",
            daemon=True,
        )
        self._hotkey_thread.start()

    @staticmethod
    def _ease_in_out(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)

    def _cursor_is_over_game_window(self, position: tuple[int, int]) -> bool:
        try:
            window = win32gui.WindowFromPoint(position)
            root = win32gui.GetAncestor(window, win32con.GA_ROOT)
            title = win32gui.GetWindowText(root)
            return self.window_title.casefold() in title.casefold()
        except Exception:
            logger.debug("Не удалось определить окно под курсором")
            return False

    def _resume_from_pause(self, reason: str) -> None:
        with self._resume_lock:
            self._resume_required = True
        self._pause_source = None
        self._pause_event.clear()
        logger.info("{}; возвращение курсора в сохранённую точку", reason)

    def _request_restart(self, reason: str) -> None:
        if self.restart_requested.is_set():
            return
        logger.warning("Запрошен перезапуск по {}", reason)
        self.restart_requested.set()
        self._pause_event.clear()
        _thread.interrupt_main()

    def _monitor_pause_hotkey(self) -> None:
        """Следит за клавишами и мышью для паузы и перезапуска."""
        multiply_was_pressed = False
        right_was_pressed = False
        middle_was_pressed = False
        last_position = tuple(int(value) for value in pydirectinput.position())
        while not self._shutdown_event.is_set():
            current_position = tuple(int(value) for value in pydirectinput.position())
            multiply_pressed = bool(
                win32api.GetAsyncKeyState(win32con.VK_MULTIPLY) & 0x8000
            )
            right_pressed = bool(
                win32api.GetAsyncKeyState(win32con.VK_RBUTTON) & 0x8000
            )
            middle_pressed = bool(
                win32api.GetAsyncKeyState(win32con.VK_MBUTTON) & 0x8000
            )

            if multiply_pressed and not multiply_was_pressed:
                alt_pressed = bool(win32api.GetAsyncKeyState(win32con.VK_MENU) & 0x8000)
                if alt_pressed:
                    self._request_restart("Alt + NUMPAD *")
                elif self._pause_event.is_set():
                    self._resume_from_pause("Пауза снята по NUMPAD *")
                else:
                    self._pause_position = current_position
                    self._pause_source = "hotkey"
                    self._pause_event.set()
                    logger.warning(
                        "Бот поставлен на паузу NUMPAD *; позиция курсора: {}",
                        self._pause_position,
                    )

            if (
                self.manual_pause_enabled
                and not self._pause_event.is_set()
                and not self._automation_mouse_event.is_set()
                and math.dist(last_position, current_position)
                >= self.mouse_jerk_threshold_px
            ):
                self._pause_position = last_position
                self._pause_source = "mouse"
                self._pause_event.set()
                logger.warning(
                    "Бот поставлен на паузу резким движением мыши: {} -> {}",
                    last_position,
                    current_position,
                )

            if (
                right_pressed
                and not right_was_pressed
                and self._pause_event.is_set()
                and self._pause_source == "mouse"
                and self._cursor_is_over_game_window(current_position)
            ):
                self._resume_from_pause("Пауза снята ПКМ по окну Miscrits")

            if (
                middle_pressed
                and not middle_was_pressed
                and self._cursor_is_over_game_window(current_position)
            ):
                self._request_restart("СКМ по окну Miscrits")

            multiply_was_pressed = multiply_pressed
            right_was_pressed = right_pressed
            middle_was_pressed = middle_pressed
            last_position = current_position
            self._shutdown_event.wait(max(0.01, self.mouse_poll_interval))

    def wait_if_paused(self) -> float:
        """Блокирует игровые действия и восстанавливает курсор после паузы."""
        started_at = time.monotonic()
        while self._pause_event.is_set() and not self._shutdown_event.is_set():
            self._shutdown_event.wait(0.08)
        paused_for = time.monotonic() - started_at

        return_position: tuple[int, int] | None = None
        with self._resume_lock:
            if self._resume_required:
                self._resume_required = False
                return_position = self._pause_position
        if return_position is not None and not self._shutdown_event.is_set():
            logger.info(
                "Плавное возвращение курсора после паузы: x={}, y={}",
                return_position[0],
                return_position[1],
            )
            self._move_mouse_human(
                return_position[0],
                return_position[1],
                honor_pause=False,
            )
        return paused_for

    def cursor_position(self) -> tuple[int, int]:
        x, y = pydirectinput.position()
        return int(x), int(y)

    def move_mouse_human(self, x: int, y: int, max_spread: float | None = None) -> None:
        """Перемещает курсор по кубической кривой Безье с микродрожанием."""
        self.wait_if_paused()
        self._move_mouse_human(x, y, honor_pause=True, max_spread=max_spread)

    def _move_mouse_human(
        self,
        x: int,
        y: int,
        honor_pause: bool,
        max_spread: float | None = None,
    ) -> None:
        self._automation_mouse_event.set()
        try:
            self._move_mouse_human_path(x, y, honor_pause, max_spread=max_spread)
        finally:
            self._automation_mouse_event.clear()

    def _move_mouse_human_path(
        self,
        x: int,
        y: int,
        honor_pause: bool,
        max_spread: float | None = None,
    ) -> None:
        start_x, start_y = pydirectinput.position()
        distance = math.hypot(x - start_x, y - start_y)
        steps = max(24, min(110, int(distance / random.uniform(5.5, 9.0))))
        spread_cap = 130.0 if max_spread is None else max(0.0, float(max_spread))
        spread = max(8.0, min(spread_cap, distance * 0.22)) if spread_cap else 0.0
        control1 = (
            start_x + (x - start_x) * random.uniform(0.20, 0.40) + random.uniform(-spread, spread),
            start_y + (y - start_y) * random.uniform(0.20, 0.40) + random.uniform(-spread, spread),
        )
        control2 = (
            start_x + (x - start_x) * random.uniform(0.60, 0.82) + random.uniform(-spread, spread),
            start_y + (y - start_y) * random.uniform(0.60, 0.82) + random.uniform(-spread, spread),
        )
        # pydirectinput может игнорировать очень короткий duration. Поэтому
        # длительность траектории обеспечивается явной паузой между точками.
        total_duration = (
            random.uniform(0.375, 0.675) + min(distance / 2000.0, 0.4)
        ) / 1.3
        step_duration = total_duration / steps
        logger.debug(
            "Наведение курсора: расстояние {:.0f}px, длительность ~{:.2f} сек",
            distance,
            total_duration,
        )

        for index in range(1, steps + 1):
            if self.restart_requested.is_set():
                return
            if honor_pause:
                self.wait_if_paused()
            t = self._ease_in_out(index / steps)
            inv = 1.0 - t
            px = (
                inv**3 * start_x
                + 3 * inv**2 * t * control1[0]
                + 3 * inv * t**2 * control2[0]
                + t**3 * x
            )
            py = (
                inv**3 * start_y
                + 3 * inv**2 * t * control1[1]
                + 3 * inv * t**2 * control2[1]
                + t**3 * y
            )
            jitter = 0 if index == steps else random.randint(1, 3)
            pydirectinput.moveTo(
                int(px + random.randint(-jitter, jitter)),
                int(py + random.randint(-jitter, jitter)),
                duration=0,
            )
            random_sleep(step_duration * 0.78, step_duration * 1.22)

    def click_human(
        self,
        x: int,
        y: int,
        *,
        apply_post_delay: bool = True,
        not_before: float | None = None,
        hold_range: tuple[float, float] | None = None,
        pre_click_delay_range: tuple[float, float] | None = None,
        not_before_reason: str = "интервал между атаками",
        not_before_wait_kind: WaitKind = "artificial",
        max_spread: float | None = None,
    ) -> tuple[int, int]:
        """Безопасно имитирует неточный человеческий клик и коррекцию."""
        try:
            if random.random() < self.miss_chance:
                radius = random.randint(2, 5)
                angle = random.uniform(0, math.tau)
                near_x = x + int(math.cos(angle) * radius)
                near_y = y + int(math.sin(angle) * radius)
                self.move_mouse_human(near_x, near_y, max_spread=max_spread)
                random_sleep(0.08, 0.22, "коррекция наведения")
            actual_x = x + random.randint(-1, 1)
            actual_y = y + random.randint(-1, 1)
            self.move_mouse_human(actual_x, actual_y, max_spread=max_spread)
            if self.restart_requested.is_set():
                current_x, current_y = pydirectinput.position()
                return int(current_x), int(current_y)
            self.wait_if_paused()
            delay_min, delay_max = pre_click_delay_range or (0.10, 0.32)
            random_sleep(
                min(delay_min, delay_max),
                max(delay_min, delay_max),
                "курсор задержался над объектом",
            )
            self.wait_if_paused()
            if not_before is not None:
                remaining = not_before - time.monotonic()
                if remaining > 0:
                    random_sleep(
                        remaining,
                        remaining,
                        not_before_reason,
                        wait_kind=not_before_wait_kind,
                    )
            self.wait_if_paused()
            if self.restart_requested.is_set():
                current_x, current_y = pydirectinput.position()
                return int(current_x), int(current_y)
            pydirectinput.moveTo(actual_x, actual_y, duration=0)
            hold_min, hold_max = hold_range or (0.07, 0.16)
            hold_duration = random.uniform(
                min(hold_min, hold_max),
                max(hold_min, hold_max),
            )
            pydirectinput.mouseDown(button="left")
            try:
                random_sleep(hold_duration, hold_duration, "удержание кнопки мыши")
            finally:
                # Не оставляем кнопку зажатой даже при прерывании работы.
                pydirectinput.mouseUp(button="left")
                self.last_click_at = time.monotonic()
            self.action_count += 1
            logger.info(
                "Клик: x={}, y={}, удержание={:.3f} сек",
                actual_x,
                actual_y,
                hold_duration,
            )
            if apply_post_delay:
                random_sleep(self.action_delay_min, self.action_delay_max, "после клика")
            return actual_x, actual_y
        except Exception:
            logger.exception("Ошибка эмуляции клика в {}, {}", x, y)
            raise

    def click_stationary_human(self, not_before: float | None = None) -> tuple[int, int]:
        """Кликает в текущей позиции, совершенно не перемещая курсор."""
        try:
            self.wait_if_paused()
            if self.restart_requested.is_set():
                current_x, current_y = pydirectinput.position()
                return int(current_x), int(current_y)
            if not_before is not None:
                remaining = not_before - time.monotonic()
                if remaining > 0:
                    random_sleep(
                        remaining,
                        remaining,
                        "интервал между атаками",
                        wait_kind="artificial",
                    )
            self.wait_if_paused()
            if self.restart_requested.is_set():
                current_x, current_y = pydirectinput.position()
                return int(current_x), int(current_y)
            actual_x, actual_y = (int(value) for value in pydirectinput.position())
            hold_duration = random.uniform(0.07, 0.16)
            pydirectinput.mouseDown(button="left")
            try:
                random_sleep(hold_duration, hold_duration, "удержание кнопки мыши")
            finally:
                pydirectinput.mouseUp(button="left")
                self.last_click_at = time.monotonic()
            self.action_count += 1
            logger.info(
                "Клик без движения: x={}, y={}, удержание={:.3f} сек",
                actual_x,
                actual_y,
                hold_duration,
            )
            return actual_x, actual_y
        except Exception:
            logger.exception("Ошибка клика без перемещения курсора")
            raise

    def close(self) -> None:
        """Останавливает монитор горячей клавиши и снимает ожидание паузы."""
        self._shutdown_event.set()
        self._pause_event.clear()
        if self._hotkey_thread.is_alive():
            self._hotkey_thread.join(timeout=0.5)
