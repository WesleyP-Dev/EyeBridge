"""Ritmo por prazo absoluto, sem espera ocupada; resolução temporária no Windows."""
import ctypes
import sys
import time


class FramePacer:
    def __init__(self, fps):
        self.period = 1/fps
        self.deadline = time.perf_counter()

    def delay(self):
        now = time.perf_counter()
        self.deadline += self.period
        if self.deadline < now:
            self.deadline = now
        return max(0, self.deadline-now)


class TimerResolution:
    def __init__(self):
        self.library = None

    def start(self):
        if sys.platform == "win32" and self.library is None:
            try:
                library = ctypes.WinDLL("winmm")
                if library.timeBeginPeriod(1) == 0:
                    self.library = library
            except OSError:
                pass
        return self

    def close(self):
        if self.library is not None:
            self.library.timeEndPeriod(1)
            self.library = None
