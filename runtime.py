from __future__ import annotations

import sys

if sys.platform == "darwin":
    from human_input_mac import HumanInput, WaitKind, random_sleep
    from human_input_mac import _button_down, _key_down, _position
    from screen_capture_mac import GameWindowCapture

    _KEY_ESCAPE = 53
    _MOUSE_LEFT = 0

    def poll_escape() -> bool:
        return _key_down(_KEY_ESCAPE)

    def poll_left_mouse() -> bool:
        return _button_down(_MOUSE_LEFT)

    def poll_f8() -> bool:
        return _key_down(100)

    def cursor_screen_position() -> tuple[int, int]:
        return _position()
else:
    import win32api
    import win32con

    from human_input import HumanInput, WaitKind, random_sleep
    from screen_capture import GameWindowCapture

    def poll_escape() -> bool:
        return bool(win32api.GetAsyncKeyState(win32con.VK_ESCAPE) & 0x8000)

    def poll_left_mouse() -> bool:
        return bool(win32api.GetAsyncKeyState(win32con.VK_LBUTTON) & 0x8000)

    def poll_f8() -> bool:
        return bool(win32api.GetAsyncKeyState(win32con.VK_F8) & 0x8000)

    def cursor_screen_position() -> tuple[int, int]:
        x, y = win32api.GetCursorPos()
        return int(x), int(y)

__all__ = [
    "GameWindowCapture",
    "HumanInput",
    "WaitKind",
    "random_sleep",
    "poll_escape",
    "poll_left_mouse",
    "poll_f8",
    "cursor_screen_position",
    "make_game_capture",
]


def make_game_capture(config: dict) -> GameWindowCapture:
    title = str(config.get("window_title", "Miscrits"))
    ref = config.get("reference_resolution")
    ref = ref if isinstance(ref, dict) else {}
    kwargs: dict = {
        "reference_width": int(ref.get("width", 1920)),
        "reference_height": int(ref.get("height", 1080)),
        "layout_fit": str(ref.get("fit", "fill")),
    }
    if sys.platform != "darwin":
        return GameWindowCapture(title, **kwargs)
    macos = config.get("macos", {}) if isinstance(config.get("macos"), dict) else {}
    kwargs["titlebar_height"] = int(macos.get("titlebar_height", 0))
    kwargs["window_owner"] = str(macos.get("window_owner", ""))
    kwargs["layout_fit"] = str(macos.get("layout_fit") or "contain")
    return GameWindowCapture(title, **kwargs)

