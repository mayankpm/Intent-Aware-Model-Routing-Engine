"""Progress bar with elapsed time and ETA for long-running scripts."""
import sys
import time


class Progress:
    """One line per step when stdout is a file (so `tail -f` / `Get-Content -Wait`
    shows it live); redraws in place on a terminal."""

    def __init__(self, total, width=30):
        self.total, self.width, self.done = total, width, 0
        self.start = time.perf_counter()
        self.tty = sys.stdout.isatty()

    @staticmethod
    def fmt(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}"

    def elapsed(self):
        return time.perf_counter() - self.start

    def step(self, label):
        self.done += 1
        elapsed = self.elapsed()
        frac = self.done / self.total
        filled = int(self.width * frac)
        eta = elapsed / self.done * (self.total - self.done)
        line = (f"[{'#' * filled}{'.' * (self.width - filled)}] {self.done}/{self.total} "
                f"{frac:6.1%} | elapsed {self.fmt(elapsed)} | eta {self.fmt(eta)} | {label}")
        print(("\r" + line.ljust(120)) if self.tty else line, end="" if self.tty else "\n", flush=True)
        if self.tty and self.done == self.total:
            print()
