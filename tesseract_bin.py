from __future__ import annotations

import shutil
import sys
from pathlib import Path


_WINDOWS_MARKERS = ("program files", "\\tesseract.exe")


def resolve_tesseract_cmd(configured: str) -> str:
    """Возвращает рабочий путь к tesseract для текущей ОС."""
    configured = (configured or "").strip()
    if sys.platform == "darwin":
        if configured and any(marker in configured.casefold() for marker in _WINDOWS_MARKERS):
            configured = ""
        for candidate in (
            configured,
            "/opt/homebrew/bin/tesseract",
            "/usr/local/bin/tesseract",
            shutil.which("tesseract") or "",
        ):
            if candidate and Path(candidate).is_file():
                return candidate
        return "tesseract"
    return configured
