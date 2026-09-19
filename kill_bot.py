from __future__ import annotations

import os
import signal
import subprocess
import sys


def find_bot_processes() -> list[tuple[int, str]]:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["pgrep", "-fl", "python"],
            capture_output=True,
            text=True,
            check=False,
        )
        found: list[tuple[int, str]] = []
        for line in result.stdout.splitlines():
            if "main.py" not in line:
                continue
            pid_text, _, command = line.partition(" ")
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            if pid == os.getpid():
                continue
            found.append((pid, command or line))
        return found

    import win32com.client

    wmi = win32com.client.GetObject("winmgmts:")
    processes = wmi.ExecQuery("SELECT ProcessId, Name, CommandLine FROM Win32_Process")
    found = []
    for process in processes:
        name = str(process.Name or "").lower()
        command = str(process.CommandLine or "")
        if name in {"python.exe", "pythonw.exe"} and "main.py" in command:
            found.append((int(process.ProcessId), command))
    return found


def kill_process(pid: int) -> None:
    if sys.platform == "darwin":
        os.kill(pid, signal.SIGTERM)
        return
    import win32api
    import win32con

    handle = win32api.OpenProcess(win32con.PROCESS_TERMINATE, False, pid)
    try:
        win32api.TerminateProcess(handle, 1)
    finally:
        win32api.CloseHandle(handle)


def main() -> int:
    bots = find_bot_processes()
    if not bots:
        print("Процессы бота не найдены")
        return 0

    stopped = 0
    for pid, command in bots:
        print(f"Завершаю PID {pid}: {command}")
        try:
            kill_process(pid)
            stopped += 1
        except Exception as exc:
            print(f"Не удалось завершить PID {pid}: {exc}", file=sys.stderr)
    print(f"Готово. Остановлено процессов: {stopped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
