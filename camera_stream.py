"""Aquisição em thread, um único slot e reconexão automática."""
import logging
import platform
from pathlib import Path
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from collections import deque
from timing import FramePacer
import cv2
import numpy as np
from metrics import StageMetrics

log = logging.getLogger(__name__)


class PicameraBridge:
    """Libcamera em outro Python; o leitor dedicado drena o pipe continuamente."""
    helper = Path(__file__).with_name("picamera_helper.py")
    system_python = "/usr/bin/python3"

    @classmethod
    def available(cls):
        if not Path(cls.system_python).exists():
            return False
        try:
            return subprocess.run([cls.system_python, str(cls.helper), "--probe"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  timeout=5).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def __init__(self, settings):
        self.s = settings
        self.timestamp = 0
        self.process = subprocess.Popen(
            [self.system_python, str(self.helper), "--width", str(settings.width),
             "--height", str(settings.height), "--fps", str(settings.camera_fps)],
            stdout=subprocess.PIPE, stderr=None, bufsize=0)

    def _read_exact(self, count):
        parts = bytearray()
        while len(parts) < count:
            chunk = self.process.stdout.read(count-len(parts))
            if not chunk:
                raise RuntimeError("Ponte Picamera2 encerrada")
            parts.extend(chunk)
        return parts

    def capture_array(self, _):
        timestamp, length = struct.unpack("!dI", self._read_exact(12))
        expected = self.s.width*self.s.height*3
        if length != expected:
            raise RuntimeError("Formato inesperado na ponte CSI")
        data = self._read_exact(length)
        self.timestamp = timestamp
        return np.frombuffer(data, np.uint8).reshape(self.s.height, self.s.width, 3)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
        self.process.stdout.close()


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    timestamp: float
    sequence: int
    demo: bool = False


class CameraCaptureWorker:
    def __init__(self, settings, source=0, backend="auto", demo=False):
        self.s, self.source, self.backend = settings, source, backend
        self._demo = demo
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._timestamps = deque(maxlen=90)
        self.reported_fps = 0.0
        self.mode = {}
        self.metrics = StageMetrics()
        self.scene_brightness = None
        self._native_usb = False
        self._stop = threading.Event()
        self._frame = None
        self._seq = 0
        self._bridge = None
        self._generation = 0
        self.status = "Inicializando câmera"
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)

    @property
    def demo(self):
        with self._lock:
            return self._demo

    def toggle_demo(self):
        with self._lock:
            self._demo = not self._demo
            self._generation += 1
            self._frame = None
            self._timestamps.clear()

    def start(self):
        self._thread.start()
        return self

    def latest(self, after=-1):
        with self._lock:
            return self._frame if self._frame is not None and self._frame.sequence > after else None

    def wait_latest(self, after, timeout=0.1):
        with self._condition:
            self._condition.wait_for(lambda: self._stop.is_set() or
                                     (self._frame is not None and self._frame.sequence > after), timeout)
            return self._frame if self._frame is not None and self._frame.sequence > after else None

    @property
    def fps(self):
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            span = self._timestamps[-1]-self._timestamps[0]
            return (len(self._timestamps)-1)/span if span > 0 else 0.0

    def _open(self):
        use_csi = self.backend == "picamera2"
        if self.backend == "auto" and platform.system() == "Linux":
            try:
                from picamera2 import Picamera2
                use_csi = bool(Picamera2.global_camera_info())
            except (ImportError, RuntimeError):
                use_csi = PicameraBridge.available()
        if use_csi:
            try:
                from picamera2 import Picamera2
            except ImportError:
                camera = PicameraBridge(self.s)
                self._bridge = camera
                return camera, True
            camera = Picamera2()
            try:

                configuration = camera.create_video_configuration(
                    main={"size": (self.s.width, self.s.height), "format": "RGB888"},
                    controls={"FrameRate": self.s.camera_fps}, buffer_count=2)
                camera.configure(configuration)
                camera.start()
                return camera, True
            except Exception:
                camera.close()
                raise
        api = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
        camera = cv2.VideoCapture(self.source, api)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError(f"Câmera USB {self.source} indisponível")
        if self.s.camera_mjpeg and not self._native_usb:
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.s.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.s.height)
        camera.set(cv2.CAP_PROP_FPS, self.s.camera_fps)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.reported_fps = camera.get(cv2.CAP_PROP_FPS)
        code = int(camera.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((code >> (8*i)) & 0xff) for i in range(4)).strip("\x00")
        self.mode = {"backend": "DirectShow" if platform.system() == "Windows" else "V4L2",
                     "fourcc": fourcc or "padrão do driver", "driver_fps": self.reported_fps,
                     "width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     "height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     "fallback_native": self._native_usb}
        log.info("Câmera USB negociada: %s", self.mode)
        return camera, False

    def _publish(self, image, demo, generation, timestamp=None):
        with self._lock:
            if generation != self._generation:
                return
            self._seq += 1
            captured = timestamp if timestamp is not None else time.perf_counter()
            self._frame = Frame(image, captured, self._seq, demo)
            self._timestamps.append(captured)
            self._condition.notify_all()

    def _run(self):
        camera = None
        csi = False
        generation = -1
        pacer = FramePacer(self.s.camera_fps)
        frames_from_open = 0
        brightness_at = 0.0
        try:
            while not self._stop.is_set():
                with self._lock:
                    demo, current = self._demo, self._generation
                if generation != current and camera is not None:
                    self._release(camera, csi)
                    camera = None
                generation = current
                if demo:
                    image = np.zeros((self.s.height, self.s.width, 3), np.uint8)
                    image[:] = (24, 31, 43)
                    center = (self.s.width//2, self.s.height//2)
                    cv2.ellipse(image, center, (110, 150), 0, 0, 360, (80, 99, 115), 2)
                    self._publish(image, True, generation)
                    self.status = "DEMO · câmera sintética"
                    self._stop.wait(pacer.delay())
                    continue
                if camera is None:
                    try:
                        camera, csi = self._open()
                        frames_from_open = 0
                        self.status = "CSI / Picamera2" if csi else "USB / OpenCV"
                        if not csi:
                            self.status += f" · {self.mode.get('fourcc', 'nativo')} · driver informa {self.reported_fps:.0f} FPS"
                    except Exception as exc:
                        self.status = f"Sem câmera · reconectando ({type(exc).__name__})"
                        log.debug("Reconexão de câmera: %s", exc)
                        self._stop.wait(1)
                        continue
                try:
                    read_start = time.perf_counter()
                    if csi:
                        image = camera.capture_array("main")
                        ok = image is not None
                    else:
                        ok, image = camera.read()
                    if not ok or image is None or not image.size:
                        raise RuntimeError("Quadro vazio")
                    self.metrics.record(capture_wait_read_ms=(time.perf_counter()-read_start)*1000)
                    frames_from_open += 1
                    if time.perf_counter()-brightness_at >= 0.5:
                        small = cv2.cvtColor(image[::8, ::8], cv2.COLOR_BGR2GRAY)
                        self.scene_brightness = round(float(small.mean()), 1)
                        brightness_at = time.perf_counter()
                    if frames_from_open == 1:
                        self.mode.update(width=image.shape[1], height=image.shape[0])
                    timestamp = camera.timestamp if isinstance(camera, PicameraBridge) else None
                    self._publish(image, False, generation, timestamp)
                except Exception as exc:
                    log.warning("Câmera desconectada: %s", exc)
                    if not csi and frames_from_open == 0 and self.s.camera_mjpeg and not self._native_usb:
                        self._native_usb = True
                        log.warning("Sem quadros com MJPEG; tentando formato nativo na mesma resolução")
                    self._release(camera, csi)
                    camera = None
                    with self._lock:
                        self._frame = None
                    self.status = "Câmera desconectada · reconectando"
                    self._stop.wait(1)
        finally:
            if camera is not None:
                self._release(camera, csi)

    @staticmethod
    def _release(camera, csi):
        try:
            camera.close() if csi else camera.release()
        except Exception:
            log.exception("Falha ao liberar câmera")

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._bridge is not None and self._bridge.process.poll() is None:
            self._bridge.process.terminate()
        self._thread.join(timeout=3)
        if self._thread.is_alive():
            log.warning("Driver de câmera bloqueado; thread daemon será finalizada pelo processo")
