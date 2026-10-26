"""Register the background data update with the OS scheduler (Windows Task Scheduler / cron)."""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ("IdxBtUpdateDaily", "IdxBtUpdateLogon")


def _cmd() -> str:
    return f'cd /d "{ROOT}" && "{sys.executable}" -m idxbt update'


def windows_commands(at: str = "17:00") -> list[list[str]]:
    run = f'cmd /c {_cmd()}'
    return [
        ["schtasks", "/Create", "/F", "/TN", TASKS[0], "/SC", "WEEKLY", "/D", "MON,TUE,WED,THU,FRI", "/ST", at, "/TR", run],
        ["schtasks", "/Create", "/F", "/TN", TASKS[1], "/SC", "ONLOGON", "/DELAY", "0001:00", "/TR", run],
    ]


def cron_lines(at: str = "17:00") -> list[str]:
    h, m = at.split(":")
    base = f'cd "{ROOT}" && "{sys.executable}" -m idxbt update >> data/update.log 2>&1'
    return [f"{int(m)} {int(h)} * * 1-5 {base}  # idxbt", f"@reboot sleep 60 && {base}  # idxbt"]


def install(at: str = "17:00") -> str:
    if platform.system() == "Windows":
        for c in windows_commands(at):
            subprocess.run(c, check=True)
        return f"Scheduled: weekdays {at} + at every login (Task Scheduler: {', '.join(TASKS)})"
    cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    keep = [l for l in cur.splitlines() if "# idxbt" not in l]
    subprocess.run(["crontab", "-"], input="\n".join(keep + cron_lines(at)) + "\n", text=True, check=True)
    return f"Scheduled in crontab: weekdays {at} + at reboot"


def remove() -> str:
    if platform.system() == "Windows":
        for t in TASKS:
            subprocess.run(["schtasks", "/Delete", "/F", "/TN", t])
        return "Removed scheduled tasks"
    cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    keep = [l for l in cur.splitlines() if "# idxbt" not in l]
    subprocess.run(["crontab", "-"], input="\n".join(keep) + "\n", text=True, check=True)
    return "Removed crontab entries"
