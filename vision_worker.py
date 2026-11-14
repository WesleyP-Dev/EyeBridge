"""Inferência independente da interface; nunca acumula quadros."""
import logging
import threading
import time
from collections import deque
from core import Observation
from detector import DemoInput, EyeDetector
from timing import FramePacer
from metrics import StageMetrics

log = logging.getLogger(__name__)


class VisionWorker:
    def __init__(self, camera, settings, automatic_demo=False, session=None):
        self.camera, self.s = camera, settings
        self.demo_input = DemoInput(automatic_demo)
        self.session = session
        self.metrics = StageMetrics()
        self.processed_frames = self.face_frames = self.gaze_frames = 0
        self._times = deque(maxlen=90)
        self._lock = threading.Lock()
        self._demo_lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = None
        self.status = "Aguardando vídeo"
        self.latency_ms = 0.0
        self.error_message = ""
        self.initialization_failure = None
        self._last_demo = None
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def latest(self):
        with self._lock:
            return self._latest

    @property
    def fps(self):
        with self._lock:
            if self.error_message or len(self._times) < 2 or time.perf_counter()-self._times[-1] > self.s.max_gap:
                return 0.0
            span = self._times[-1]-self._times[0]
            return (len(self._times)-1)/span if span > 0 else 0.0

    def move_demo(self, x, y=None):
        with self._demo_lock:
            self.demo_input.x = x
            if y is not None:
                self.demo_input.y = y

    def blink_demo(self):
        with self._demo_lock:
            self.demo_input.blink(time.perf_counter())

    def _run(self):
        detector = None
        sequence = -1
        retry_at = 0
        pacer = FramePacer(self.s.processing_fps)
        try:
            while not self._stop.is_set():
                frame = self.camera.wait_latest(sequence)
                if self.session:
                    self.session.idle()
                if frame is not None:
                    start = time.perf_counter()
                    sequence = frame.sequence
                    if frame.demo != self._last_demo:
                        with self._lock:
                            self._times.clear()
                        self._last_demo = frame.demo
                    try:
                        if frame.demo:
                            with self._demo_lock:
                                obs = self.demo_input.observation(frame.timestamp)
                            self.status = "DEMO · olhar simulado"
                        else:
                            if detector is None:
                                if self.initialization_failure is not None:
                                    self.error_message = self.initialization_failure
                                    self.status = "ERRO NO DETECTOR · reinicie e execute --check"
                                    self._stop.wait(0.1)
                                    continue
                                if start < retry_at:
                                    self._stop.wait(0.1)
                                    continue
                                detector = EyeDetector()
                            obs = detector.process(frame.image, frame.timestamp)
                            self.status = ("Olhar consistente" if obs.gaze_valid else "Ajuste a posição: olhos com leituras divergentes") if obs.valid else "Posicione o rosto à frente da câmera"
                        self.latency_ms = (time.perf_counter()-start)*1000
                        self.error_message = ""
                        group = "demo_ms" if frame.demo else ("inference_with_face_ms" if obs.valid else "inference_without_face_ms")
                        self.metrics.record(inference_ms=self.latency_ms,
                                            **{group: self.latency_ms},
                                            frame_age_ms=(time.perf_counter()-frame.timestamp)*1000)
                        if detector is not None and not frame.demo:
                            self.metrics.record(**detector.timings)
                        control_start = time.perf_counter()
                        if self.session:
                            self.session.consume(frame, obs)
                        self.metrics.record(control_ms=(time.perf_counter()-control_start)*1000)
                        with self._lock:
                            self._times.append(time.perf_counter())
                            self.processed_frames += 1
                            self.face_frames += int(obs.valid and not frame.demo)
                            self.gaze_frames += int(obs.valid and obs.gaze_valid and not frame.demo)
                            self._latest = (frame, obs)
                    except Exception as exc:
                        log.exception("Falha no detector")
                        self.error_message = f"{type(exc).__name__}: {exc}"
                        self.status = "ERRO NO DETECTOR · execute --check"
                        if detector is None:


                            self.initialization_failure = self.error_message
                        with self._lock:
                            self._times.clear()
                            self._latest = (frame, Observation(frame.timestamp))
                        if self.session:
                            self.session.consume(frame, Observation(frame.timestamp))
                            self.session.detector_failed("Detector indisponível. Reinicie e execute --check.")
                        if detector is not None:
                            detector.close()
                            detector = None
                        retry_at = start+5
                self._stop.wait(pacer.delay())
        finally:
            if detector is not None:
                detector.close()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3)
