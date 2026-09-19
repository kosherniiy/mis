from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from behavior import MiscritsBehavior


def parse_attack_sequence(value: str) -> list[int]:
    try:
        sequence = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "атаки должны быть числами от 1 до 4 через запятую"
        ) from exc
    if not sequence or any(attack not in (1, 2, 3, 4) for attack in sequence):
        raise argparse.ArgumentTypeError(
            "последовательность должна содержать номера атак 1–4"
        )
    return sequence


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Конфиг не найден: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Корень config.yaml должен быть YAML-объектом")
    return config


def configure_logging(debug: bool) -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if debug else "INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | <level>{message}</level>",
    )
    logger.add(
        "logs/bot_{time:YYYY-MM-DD}.log",
        level="INFO",
        rotation="00:00",
        retention="14 days",
        encoding="utf-8",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Автоматизация боёв Miscrits")
    parser.add_argument(
        "--attack",
        type=parse_attack_sequence,
        help=(
            "Циклическая последовательность атак через запятую, например 1,2,2. "
            "Переопределяет attack_sequence из config.yaml."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=("search", "lvl_up_continue", "character_promotion"),
        help="Стартовая фаза. Переопределяет start_phase из config.yaml.",
    )
    parser.add_argument(
        "--notifyStart",
        "--notify-start",
        dest="notify_start",
        action="store_true",
        help="Отправить уведомление ntfy об успешном запуске бота.",
    )
    parser.add_argument(
        "--autoCatch",
        dest="auto_catch",
        action="store_true",
        help="Ловить common/rare с шансами из stop_capture_chances вместо паузы.",
    )
    parser.add_argument(
        "--autoCatchRares",
        dest="auto_catch_rares",
        action="store_true",
        help="Ловить каждого редкого мискрита.",
    )
    parser.add_argument(
        "--autoCatchAll",
        dest="auto_catch_all",
        action="store_true",
        help="Пытаться ловить каждого встречного мискрита.",
    )
    parser.add_argument(
        "--lowLevelEncounters",
        dest="low_level_encounters",
        action="store_true",
        help="В автоловле начинать со 2-го мувсета вместо 1-го.",
    )
    parser.add_argument(
        "--debugEncounters",
        dest="debug_encounters",
        action="store_true",
        help="Сохранять скрин каждого боя в папку encounters.",
    )
    parser.add_argument(
        "--stallAnyRarity",
        dest="stall_any_rarity",
        action="store_true",
        help=(
            "Для теста: на любой редкости ждать 2 минуты и жать атаку 3 "
            "1-го мувсета, пока игрок не поставит паузу."
        ),
    )
    args = parser.parse_args()

    try:
        config = load_config()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Ошибка загрузки config.yaml: {exc}", file=sys.stderr)
        return 2
    if args.attack is not None:
        config["attack_sequence"] = args.attack
    if args.phase is not None:
        config["start_phase"] = args.phase
    if args.auto_catch:
        config["autoCatch"] = True
    if args.auto_catch_all:
        config["autoCatchAll"] = True
    if args.auto_catch_rares:
        config["autoCatchRares"] = True
    if args.low_level_encounters:
        config["lowLevelEncounters"] = True
    if args.debug_encounters:
        config["debugEncounters"] = True
    if args.stall_any_rarity:
        config["stallAnyRarity"] = True

    configure_logging(bool(config.get("debug", False)))
    stop_event = threading.Event()
    user_stop_event = threading.Event()

    def request_stop(signum: int, _frame: object) -> None:
        logger.warning("Получен сигнал {}, выполняется graceful shutdown", signum)
        user_stop_event.set()
        stop_event.set()
        # Прерываем в том числе длинный сон или ERROR_PAUSE.
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    logger.info(
        "Старт бота Miscrits; платформа={}, последовательность атак={}, стартовая фаза={}, "
        "autoCatch={}, autoCatchAll={}, autoCatchRares={}, "
        "lowLevelEncounters={}, stallAnyRarity={}",
        sys.platform,
        config.get("attack_sequence", [config.get("selected_attack", 2)]),
        config.get("start_phase", "search"),
        bool(config.get("autoCatch", False)),
        bool(config.get("autoCatchAll", False)),
        bool(config.get("autoCatchRares", False)),
        bool(config.get("lowLevelEncounters", False)),
        bool(config.get("stallAnyRarity", False)),
    )
    if sys.platform == "darwin":
        logger.info(
            "macOS: разреши Accessibility и Screen Recording для Terminal/Python. "
            "Fullscreen на отдельном столе поддерживается — бот сам переключит Space. "
            "Пауза: F8. Перезапуск: Option+F8 или СКМ."
        )
    bot = MiscritsBehavior(config, stop_event)
    if args.notify_start:
        bot.notifier.send(
            (
                "Бот запущен. Последовательность атак: "
                f"{config.get('attack_sequence', [config.get('selected_attack', 2)])}, "
                f"стартовая фаза: {config.get('start_phase', 'search')}."
            ),
            title="Miscrits bot запущен",
            priority=3,
            tags=["white_check_mark", "video_game"],
        )
    restart_requested = False
    try:
        bot.run()
        restart_requested = bot.input.restart_requested.is_set()
    except KeyboardInterrupt:
        restart_requested = bot.input.restart_requested.is_set()
        if not restart_requested:
            user_stop_event.set()
        stop_event.set()
    except Exception:
        logger.exception("Критическая ошибка верхнего уровня")
        bot.notifier.send(
            "Бот завершает работу из-за критической ошибки верхнего уровня.",
            title="Критическая ошибка Miscrits",
            priority=5,
            tags=["rotating_light", "warning"],
        )
        bot.save_state()
        return 1
    finally:
        stop_event.set()
        if restart_requested:
            logger.warning("Перезапуск бота по Alt + NUMPAD *")
        elif user_stop_event.is_set():
            logger.info("Бот остановлен пользователем")
        else:
            logger.info("Бот завершил работу")
        logger.complete()
    if restart_requested:
        os.execv(sys.executable, [sys.executable, *sys.argv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
