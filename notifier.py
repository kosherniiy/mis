from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any

from loguru import logger


class NtfyNotifier:
    """Отправляет уведомления ntfy в фоне, не блокируя игровой цикл."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.server = str(config.get("server", "https://ntfy.sh")).rstrip("/")
        self.topic = str(config.get("topic", "")).strip()
        self.timeout = max(0.5, float(config.get("timeout", 10.0)))

    def send(
        self,
        message: str,
        *,
        title: str = "Miscrits bot",
        priority: int = 4,
        tags: list[str] | None = None,
    ) -> bool:
        if not self.enabled or not self.topic:
            return False

        thread = threading.Thread(
            target=self._send_blocking,
            args=(message, title, priority, tags or []),
            name="ntfy-send",
            daemon=True,
        )
        thread.start()
        return True

    def _send_blocking(
        self,
        message: str,
        title: str,
        priority: int,
        tags: list[str],
    ) -> None:
        payload = json.dumps(
            {
                "topic": self.topic,
                "message": message,
                "title": title,
                "priority": priority,
                "tags": tags,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.server,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response.read()
            logger.info("Уведомление ntfy отправлено в тему {}", self.topic)
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            logger.warning(
                "Не удалось отправить уведомление ntfy в {}: {}",
                self.topic,
                exc,
            )
