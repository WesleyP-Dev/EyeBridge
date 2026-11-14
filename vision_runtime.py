"""Carrega o MediaPipe uma única vez."""
import importlib
import threading


class VisionRuntimeError(RuntimeError):
    pass


class RuntimeLoader:
    def __init__(self):
        self.module = None
        self.failure = None
        self.lock = threading.Lock()

    def load(self):
        with self.lock:
            if self.failure is not None:
                raise VisionRuntimeError(self.describe(self.failure)) from self.failure
            if self.module is None:
                try:
                    module = importlib.import_module("mediapipe")
                    if not callable(module.solutions.face_mesh.FaceMesh):
                        raise ImportError("FaceMesh não está disponível neste pacote MediaPipe")
                    self.module = module
                except Exception as exc:
                    self.failure = exc
                    raise VisionRuntimeError(self.describe(exc)) from exc
            return self.module

    @staticmethod
    def describe(exc):
        return f"Falha ao carregar MediaPipe: {type(exc).__name__}: {exc}. Reinicie e execute main.py --check --verbose."


runtime = RuntimeLoader()
