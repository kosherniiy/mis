from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import yaml

from runtime import (
    cursor_screen_position,
    make_game_capture,
    poll_escape,
    poll_f8,
    poll_left_mouse,
)


SAMPLE_INTERVAL = 0.004
OUTPUT_DIR = Path("debug") / "mouse_traces"


def _summarize(samples: list[dict[str, object]]) -> dict[str, object]:
    if len(samples) < 2:
        return {"error": "слишком мало точек"}

    path_px = 0.0
    max_speed = 0.0
    outside = 0
    clicks: list[dict[str, object]] = []
    down_index: int | None = None

    for index, sample in enumerate(samples):
        if not bool(sample["inside"]):
            outside += 1
        if index == 0:
            continue
        prev = samples[index - 1]
        dx = int(sample["sx"]) - int(prev["sx"])
        dy = int(sample["sy"]) - int(prev["sy"])
        dist = (dx * dx + dy * dy) ** 0.5
        dt = float(sample["t"]) - float(prev["t"])
        path_px += dist
        if dt > 0:
            max_speed = max(max_speed, dist / dt)

        was_down = bool(prev["lb"])
        is_down = bool(sample["lb"])
        if is_down and not was_down:
            down_index = index
        elif was_down and not is_down and down_index is not None:
            start = samples[down_index]
            drag = 0.0
            for drag_index in range(down_index + 1, index + 1):
                a = samples[drag_index - 1]
                b = samples[drag_index]
                ddx = int(b["sx"]) - int(a["sx"])
                ddy = int(b["sy"]) - int(a["sy"])
                drag += (ddx * ddx + ddy * ddy) ** 0.5
            hover = 0.0
            cursor_x, cursor_y = int(start["sx"]), int(start["sy"])
            for lookback in range(down_index, 0, -1):
                point = samples[lookback - 1]
                if (
                    abs(int(point["sx"]) - cursor_x) > 4
                    or abs(int(point["sy"]) - cursor_y) > 4
                ):
                    hover = float(start["t"]) - float(point["t"])
                    break
            clicks.append(
                {
                    "down_t": round(float(start["t"]), 4),
                    "up_t": round(float(sample["t"]), 4),
                    "hold_ms": round((float(sample["t"]) - float(start["t"])) * 1000, 1),
                    "local": [int(start["lx"]), int(start["ly"])],
                    "screen": [int(start["sx"]), int(start["sy"])],
                    "drag_px": round(drag, 1),
                    "hover_before_ms": round(hover * 1000, 1),
                }
            )
            down_index = None

    first, last = samples[0], samples[-1]
    duration = float(last["t"]) - float(first["t"])
    straight = (
        (int(last["sx"]) - int(first["sx"])) ** 2
        + (int(last["sy"]) - int(first["sy"])) ** 2
    ) ** 0.5
    return {
        "duration_s": round(duration, 3),
        "samples": len(samples),
        "path_px": round(path_px, 1),
        "straight_px": round(straight, 1),
        "straightness": round(straight / path_px, 3) if path_px else 0.0,
        "max_speed_px_s": round(max_speed, 1),
        "avg_speed_px_s": round(path_px / duration, 1) if duration else 0.0,
        "outside_window_samples": outside,
        "start_local": [int(first["lx"]), int(first["ly"])],
        "end_local": [int(last["lx"]), int(last["ly"])],
        "clicks": clicks,
    }


def main() -> int:
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    capture = make_game_capture(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Запись движения мыши для сравнения с ботом.")
    print("1) Поставь бота на паузу (NUMPAD *) или останови его.")
    print("2) Открой окно прокачки, наведись к «Прокачать».")
    print("3) F8 — старт, затем как обычно: Прокачать → Далее.")
    print("4) F8 — стоп (или ESC — выход).")
    print()

    recording = False
    samples: list[dict[str, object]] = []
    started_at = 0.0
    f8_was = False
    try:
        while True:
            f8 = poll_f8()
            if poll_escape():
                print("Выход.", flush=True)
                return 0
            if f8 and not f8_was:
                if not recording:
                    samples = []
                    started_at = time.perf_counter()
                    recording = True
                    print("Запись началась…", flush=True)
                else:
                    recording = False
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = OUTPUT_DIR / f"human_{stamp}.json"
                    summary = _summarize(samples)
                    payload = {"kind": "human", "created": stamp, "summary": summary, "samples": samples}
                    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"Сохранено: {path.resolve()}", flush=True)
                    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
                    print("Можно записывать ещё раз (F8) или выйти (ESC).", flush=True)
            f8_was = f8

            if recording:
                screen_x, screen_y = cursor_screen_position()
                try:
                    region = capture.get_region()
                except Exception:
                    region = None
                if region is None:
                    local_x = local_y = -1
                    inside = False
                else:
                    local_x = screen_x - region.left
                    local_y = screen_y - region.top
                    inside = 0 <= local_x < region.width and 0 <= local_y < region.height
                samples.append(
                    {
                        "t": round(time.perf_counter() - started_at, 4),
                        "sx": screen_x,
                        "sy": screen_y,
                        "lx": local_x,
                        "ly": local_y,
                        "lb": poll_left_mouse(),
                        "inside": inside,
                    }
                )
            time.sleep(SAMPLE_INTERVAL)
    finally:
        capture.close()


if __name__ == "__main__":
    raise SystemExit(main())
