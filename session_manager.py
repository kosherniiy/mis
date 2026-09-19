from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class SessionManager:
    """Сохраняет и восстанавливает состояние бота."""

    def __init__(self, config: dict[str, Any], state_path: str | Path = "state.json") -> None:
        self.config = config
        self.state_path = Path(state_path)
        self.state: dict[str, Any] = self.load_state()

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            logger.exception("Не удалось загрузить {}", self.state_path)
            return {}

    def save_state(self, runtime: dict[str, Any]) -> None:
        payload = {
            **self.state,
            **runtime,
            "work_until": None,
            "sleep_until": None,
            "last_sleep_day": None,
            "saved_at": datetime.now().isoformat(),
        }
        try:
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.state_path)
            self.state = payload
            logger.info("Состояние сохранено: {}", self.state_path)
        except OSError:
            logger.exception("Не удалось сохранить состояние")
