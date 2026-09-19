from __future__ import annotations

import _thread
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

try:
    from Quartz import (
        CGAssociateMouseAndMouseCursorPosition,
        CGEventCreate,
        CGEventCreateMouseEvent,
        CGEventGetLocation,
        CGEventPost,
        CGEventSourceButtonState,
        CGEventSourceCreate,
        CGEventSourceKeyState,
        CGEventSourceSetLocalEventsFilterDuringSuppressionState,
        CGEventSourceSetLocalEventsSuppressionInterval,
        CGPoint,
        CGWarpMouseCursorPosition,
        kCGEventFilterMaskPermitLocalKeyboardEvents,
        kCGEventFilterMaskPermitLocalMouseEvents,
        kCGEventFilterMaskPermitSystemDefinedEvents,
        kCGEventLeftMouseDown,
        kCGEventLeftMouseUp,
        kCGEventMouseMoved,
        kCGEventSourceStateCombinedSessionState,
        kCGEventSourceStateHIDSystemState,
        kCGEventSuppressionStateRemoteMouseDrag,
        kCGEventSuppressionStateSuppressionInterval,
        kCGHIDEventTap,
        kCGMouseButtonLeft,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Для macOS нужен pyobjc-framework-Quartz. "
        "Установи зависимости из requirements-mac.txt"
    ) from exc

try:
    from Quartz import CGSetLocalEventsSuppressionInterval
except ImportError:  # pragma: no cover
    CGSetLocalEventsSuppressionInterval = None


WaitKind = Literal["artificial", "forced"]

# HID keycodes: F8, Option, keypad *
_KEY_F8 = 100
_KEY_OPTION = 58
_KEY_KEYPAD_MULTIPLY = 67
_MOUSE_LEFT = 0
_MOUSE_RIGHT = 1
_MOUSE_MIDDLE = 2

# CombinedSessionState + interval 0 снимают 0.25с freeze после warp.
# Иначе CGWarpMouseCursorPosition глушит ввод, и курсор ползёт рывками.
_EVENT_SOURCE = CGEventSourceCreate(kCGEventSourceStateCombinedSessionState)
_PERMIT_LOCAL_EVENTS = (
    kCGEventFilterMaskPermitLocalMouseEvents
    | kCGEventFilterMaskPermitLocalKeyboardEvents
    | kCGEventFilterMaskPermitSystemDefinedEvents
)


def _disable_warp_suppression() -> None:
    """Warp без этого игнорирует локальную мышь ~0.25с после каждого шага."""
    if CGSetLocalEventsSuppressionInterval is not None:
        try:
            CGSetLocalEventsSuppressionInterval(0.0)
        except Exception:
            pass
    if _EVENT_SOURCE is not None:
        CGEventSourceSetLocalEventsSuppressionInterval(_EVENT_SOURCE, 0.0)


def _configure_event_source() -> None:
    _disable_warp_suppression()
    if _EVENT_SOURCE is None:
        return
    CGEventSourceSetLocalEventsFilterDuringSuppressionState(
        _EVENT_SOURCE,
        _PERMIT_LOCAL_EVENTS,
        kCGEventSuppressionStateSuppressionInterval,
    )
    CGEventSourceSetLocalEventsFilterDuringSuppressionState(
        _EVENT_SOURCE,
        _PERMIT_LOCAL_EVENTS,
        kCGEventSuppressionStateRemoteMouseDrag,
    )


_configure_event_source()


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


def _key_down(code: int) -> bool:
    return bool(CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, code))


def _button_down(button: int) -> bool:
    return bool(CGEventSourceButtonState(kCGEventSourceStateHIDSystemState, button))


# Движение на macOS строится с нуля: не копируем Windows.
# Warp прыгает в точку мгновенно, поэтому шаг должен быть < 1px, иначе это телепорт.
_STEP_PX = 0.5
_AVG_SPEED_PX_S = 400.0
_MIN_MOVE_S = 0.32
_MAX_MOVE_S = 1.8


def _position() -> tuple[int, int]:
    loc = CGEventGetLocation(CGEventCreate(None))
    return int(loc.x), int(loc.y)


def _position_f() -> tuple[float, float]:
    loc = CGEventGetLocation(CGEventCreate(None))
    return float(loc.x), float(loc.y)


def _wait_until(deadline: float) -> None:
    """Спин почти всего интервала: sleep на macOS склеивает соседние шаги в рывок."""
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        return
    if remaining > 0.01:
        time.sleep(remaining - 0.006)
    while time.perf_counter() < deadline:
        pass


def _bezier_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    u = 1.0 - t
    uu, tt = u * u, t * t
    uuu, ttt = uu * u, tt * t
    return (
        uuu * p0[0] + 3.0 * uu * t * p1[0] + 3.0 * u * tt * p2[0] + ttt * p3[0],
        uuu * p0[1] + 3.0 * uu * t * p1[1] + 3.0 * u * tt * p2[1] + ttt * p3[1],
    )


def _speed_profile(u: float) -> float:
    """Медленнее на концах, но никогда не останавливается."""
    u = min(1.0, max(0.0, u))
    return 0.55 + 0.45 * (4.0 * u * (1.0 - u))


def _arc_path(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    spacing: float,
) -> list[tuple[float, float]]:
    chord = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
    samples = min(1800, max(80, int(chord * 3.0) + 80))
    raw: list[tuple[float, float]] = [p0]
    lengths = [0.0]
    total = 0.0
    prev = p0
    for index in range(1, samples + 1):
        point = _bezier_point(p0, p1, p2, p3, index / samples)
        total += math.hypot(point[0] - prev[0], point[1] - prev[1])
        raw.append(point)
        lengths.append(total)
        prev = point
    if total <= spacing:
        return [p0, p3]
    count = max(2, int(math.ceil(total / spacing)) + 1)
    path: list[tuple[float, float]] = []
    cursor = 1
    for index in range(count):
        wanted = total * index / (count - 1)
        while cursor < len(lengths) - 1 and lengths[cursor] < wanted:
            cursor += 1
        left = cursor - 1
        span = lengths[cursor] - lengths[left]
        if span <= 1e-9:
            path.append(raw[cursor])
            continue
        mix = (wanted - lengths[left]) / span
        ax, ay = raw[left]
        bx, by = raw[cursor]
        path.append((ax + (bx - ax) * mix, ay + (by - ay) * mix))
    path[0] = p0
    path[-1] = p3
    return path


def _warp(x: float, y: float) -> None:
    CGWarpMouseCursorPosition(CGPoint(float(x), float(y)))


def _move_to(x: float, y: float, *, announce: bool = False) -> None:
    """Ставит курсор warp'ом. CGEventPost по пути слипается в рывки из‑за coalescing."""
    point = CGPoint(float(x), float(y))
    _disable_warp_suppression()
    CGWarpMouseCursorPosition(point)
    if not announce:
        return
    event = CGEventCreateMouseEvent(
        _EVENT_SOURCE,
        kCGEventMouseMoved,
        point,
        kCGMouseButtonLeft,
    )
    CGEventPost(kCGHIDEventTap, event)


def _sync_cursor() -> None:
    CGAssociateMouseAndMouseCursorPosition(True)


def _left_down() -> None:
    _disable_warp_suppression()
    point = CGPoint(*_position())
    event = CGEventCreateMouseEvent(
        _EVENT_SOURCE,
        kCGEventLeftMouseDown,
        point,
        kCGMouseButtonLeft,
    )
    CGEventPost(kCGHIDEventTap, event)


def _left_up() -> None:
    _disable_warp_suppression()
    point = CGPoint(*_position())
    event = CGEventCreateMouseEvent(
        _EVENT_SOURCE,
        kCGEventLeftMouseUp,
        point,
        kCGMouseButtonLeft,
    )
    CGEventPost(kCGHIDEventTap, event)


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
    _window_checker: object | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.action_count = 0
        logger.info(
            "macOS ввод: пауза F8 или NUMPAD *, перезапуск Option+F8 или СКМ. "
            "Нужны разрешения Accessibility и Screen Recording."
        )
        self._hotkey_thread = threading.Thread(
            target=self._monitor_pause_hotkey,
            name="bot-control-monitor",
            daemon=True,
        )
        self._hotkey_thread.start()

    def bind_window_checker(self, checker: object) -> None:
        """Подключает проверку «курсор над окном игры» из GameWindowCapture."""
        self._window_checker = checker

    @staticmethod
    def _ease_in_out(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)

    def _cursor_is_over_game_window(self, position: tuple[int, int]) -> bool:
        checker = self._window_checker
        contains = getattr(checker, "contains_point", None)
        if callable(contains):
            try:
                return bool(contains(position[0], position[1]))
            except Exception:
                logger.debug("Не удалось определить окно под курсором")
                return False
        return True

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

    def _hotkey_pressed(self) -> bool:
        return _key_down(_KEY_F8) or _key_down(_KEY_KEYPAD_MULTIPLY)

    def _monitor_pause_hotkey(self) -> None:
        """Следит за клавишами и мышью для паузы и перезапуска."""
        hotkey_was_pressed = False
        right_was_pressed = False
        middle_was_pressed = False
        last_position = _position()
        while not self._shutdown_event.is_set():
            current_position = _position()
            hotkey_pressed = self._hotkey_pressed()
            right_pressed = _button_down(_MOUSE_RIGHT)
            middle_pressed = _button_down(_MOUSE_MIDDLE)

            if hotkey_pressed and not hotkey_was_pressed:
                option_pressed = _key_down(_KEY_OPTION)
                if option_pressed:
                    self._request_restart("Option + F8")
                elif self._pause_event.is_set():
                    self._resume_from_pause("Пауза снята по F8 / NUMPAD *")
                else:
                    self._pause_position = current_position
                    self._pause_source = "hotkey"
                    self._pause_event.set()
                    logger.warning(
                        "Бот поставлен на паузу F8 / NUMPAD *; позиция курсора: {}",
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

            hotkey_was_pressed = hotkey_pressed
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
        return _position()

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
        start = _position_f()
        target = (float(x), float(y))
        distance = math.hypot(target[0] - start[0], target[1] - start[1])
        if distance < 1.5:
            _move_to(target[0], target[1], announce=True)
            _sync_cursor()
            return

        spread_cap = 36.0 if max_spread is None else max(0.0, float(max_spread))
        spread = min(spread_cap, max(3.0, distance * 0.08)) if spread_cap else 0.0
        control1 = (
            start[0] + (target[0] - start[0]) * random.uniform(0.28, 0.38)
            + random.uniform(-spread, spread),
            start[1] + (target[1] - start[1]) * random.uniform(0.28, 0.38)
            + random.uniform(-spread, spread),
        )
        control2 = (
            start[0] + (target[0] - start[0]) * random.uniform(0.62, 0.74)
            + random.uniform(-spread, spread),
            start[1] + (target[1] - start[1]) * random.uniform(0.62, 0.74)
            + random.uniform(-spread, spread),
        )
        path = _arc_path(start, control1, control2, target, _STEP_PX)
        steps = max(1, len(path) - 1)
        duration = min(
            _MAX_MOVE_S,
            max(_MIN_MOVE_S, (steps * _STEP_PX) / _AVG_SPEED_PX_S),
        )
        weights = [_speed_profile(index / steps) for index in range(1, steps + 1)]
        inv_sum = sum(1.0 / speed for speed in weights)
        logger.debug(
            "Наведение курсора: расстояние {:.0f}px, длительность ~{:.2f} сек, шагов {}",
            distance,
            duration,
            steps,
        )

        _disable_warp_suppression()
        _sync_cursor()
        due = time.perf_counter()
        try:
            for index, ((px, py), speed) in enumerate(zip(path[1:], weights), start=1):
                if self.restart_requested.is_set():
                    return
                if honor_pause and index % 48 == 0:
                    paused_for = self.wait_if_paused()
                    _disable_warp_suppression()
                    if paused_for > 0.02:
                        due = time.perf_counter()
                _warp(px, py)
                due += duration * ((1.0 / speed) / inv_sum)
                _wait_until(due)
            _move_to(target[0], target[1], announce=True)
        finally:
            _sync_cursor()

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
                return _position()
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
                return _position()
            _move_to(actual_x, actual_y, announce=True)
            hold_min, hold_max = hold_range or (0.07, 0.16)
            hold_duration = random.uniform(
                min(hold_min, hold_max),
                max(hold_min, hold_max),
            )
            _left_down()
            try:
                random_sleep(hold_duration, hold_duration, "удержание кнопки мыши")
            finally:
                _left_up()
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
                return _position()
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
                return _position()
            actual_x, actual_y = _position()
            hold_duration = random.uniform(0.07, 0.16)
            _left_down()
            try:
                random_sleep(hold_duration, hold_duration, "удержание кнопки мыши")
            finally:
                _left_up()
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
