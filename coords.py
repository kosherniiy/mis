from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger

Kind = Literal["pixel", "rect"]
MARKER_PREFIX = "level_marker_"


@dataclass
class CoordObject:
    number: int
    key: str
    name: str
    kind: Kind
    x: int = 0
    y: int = 0
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0
    margin: int | None = None

    def as_yaml(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "key": self.key,
            "name": self.name,
            "kind": self.kind,
        }
        if self.kind == "pixel":
            data["x"] = int(self.x)
            data["y"] = int(self.y)
        else:
            data["left"] = int(self.left)
            data["top"] = int(self.top)
            data["right"] = int(self.right)
            data["bottom"] = int(self.bottom)
            if self.margin is not None:
                data["margin"] = int(self.margin)
        return data


@dataclass
class CoordBook:
    directory: Path
    width: int = 0
    height: int = 0
    objects: dict[int, CoordObject] = field(default_factory=dict)
    path: Path | None = None

    @property
    def size_key(self) -> str:
        return f"{self.width}x{self.height}"

    def object_by_key(self, key: str) -> CoordObject | None:
        for item in self.objects.values():
            if item.key == key:
                return item
        return None

    def load(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.path = self.directory / f"{self.size_key}.yaml"
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Нет координат для {self.size_key}: создайте {self.path} "
                f"через python calibrate.py или скопируйте layouts/1920x1009.yaml"
            )
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        items = raw.get("objects", {})
        if not isinstance(items, dict):
            raise ValueError(f"{self.path}: objects должен быть словарём номеров")
        loaded: dict[int, CoordObject] = {}
        for number, payload in items.items():
            obj = _parse_object(int(number), payload)
            loaded[obj.number] = obj
        if not loaded:
            raise ValueError(f"{self.path}: пустой список objects")
        self.objects = loaded
        logger.info("Координаты {} ({} объектов): {}", self.size_key, len(loaded), self.path)

    def save(self) -> Path:
        if self.path is None:
            self.path = self.directory / f"{self.size_key}.yaml"
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "resolution": {"width": self.width, "height": self.height},
            "objects": {
                number: self.objects[number].as_yaml()
                for number in sorted(self.objects)
            },
        }
        text = yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
        self.path.write_text(text, encoding="utf-8")
        logger.info("Координаты сохранены: {}", self.path)
        return self.path

    def apply_to_config(self, config: dict[str, Any]) -> None:
        """Подставляет x/y и прямоугольники в ключи config.yaml."""
        markers: list[tuple[int, int, int]] = []
        for obj in self.objects.values():
            if obj.key.startswith(MARKER_PREFIX):
                try:
                    index = int(obj.key.removeprefix(MARKER_PREFIX))
                except ValueError:
                    continue
                markers.append((index, obj.x, obj.y))
                continue
            slot = config.setdefault(obj.key, {})
            if not isinstance(slot, dict):
                slot = {}
                config[obj.key] = slot
            if obj.kind == "pixel":
                slot["x"] = int(obj.x)
                slot["y"] = int(obj.y)
            else:
                slot["left"] = int(obj.left)
                slot["top"] = int(obj.top)
                slot["right"] = int(obj.right)
                slot["bottom"] = int(obj.bottom)
                if obj.margin is not None:
                    slot["margin"] = int(obj.margin)
        if markers:
            level = config.setdefault("level_up", {})
            if not isinstance(level, dict):
                level = {}
                config["level_up"] = level
            level["marker_pixels"] = [
                {"x": x, "y": y}
                for _, x, y in sorted(markers, key=lambda item: item[0])
            ]

    def set_pixel(self, number: int, x: int, y: int) -> CoordObject:
        obj = self.objects[number]
        if obj.kind != "pixel":
            raise ValueError(f"объект {number} не пиксель")
        obj.x, obj.y = int(x), int(y)
        return obj

    def set_corner(self, number: int, which: str, x: int, y: int) -> CoordObject:
        obj = self.objects[number]
        if obj.kind != "rect":
            raise ValueError(f"объект {number} не зона")
        if which == "tl":
            obj.left, obj.top = int(x), int(y)
        else:
            obj.right, obj.bottom = int(x), int(y)
        if obj.right <= obj.left:
            obj.right = obj.left + 1
        if obj.bottom <= obj.top:
            obj.bottom = obj.top + 1
        return obj


def _parse_object(number: int, payload: object) -> CoordObject:
    if not isinstance(payload, dict):
        raise ValueError(f"объект {number}: ожидался словарь")
    kind = str(payload.get("kind", "")).strip().lower()
    if kind not in {"pixel", "rect"}:
        raise ValueError(f"объект {number}: kind должен быть pixel или rect")
    obj = CoordObject(
        number=number,
        key=str(payload.get("key", f"object_{number}")),
        name=str(payload.get("name", payload.get("key", number))),
        kind=kind,  # type: ignore[arg-type]
    )
    if kind == "pixel":
        obj.x = int(payload["x"])
        obj.y = int(payload["y"])
    else:
        obj.left = int(payload["left"])
        obj.top = int(payload["top"])
        obj.right = int(payload["right"])
        obj.bottom = int(payload["bottom"])
        if payload.get("margin") is not None:
            obj.margin = int(payload["margin"])
    return obj
