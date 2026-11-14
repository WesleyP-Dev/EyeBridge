"""Medições limitadas em memória, sem frames ou dados biométricos."""
from collections import deque
import statistics
import threading


class StageMetrics:
    def __init__(self, limit=600):
        self.limit = limit
        self.samples = {}
        self._lock = threading.Lock()

    def record(self, **values):
        with self._lock:
            for name, value in values.items():
                if name not in self.samples:
                    self.samples[name] = deque(maxlen=self.limit)
                self.samples[name].append(float(value))

    def summary(self):
        with self._lock:
            copied = {name: sorted(values) for name, values in self.samples.items() if values}
        return {name: {"samples": len(values), "median": round(statistics.median(values), 3),
                       "p95": round(values[int((len(values)-1)*0.95)], 3)}
                for name, values in copied.items()}
