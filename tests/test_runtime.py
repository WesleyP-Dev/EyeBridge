import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from config import Settings
from core import Calibration
from interaction import InteractionSession
from vision_runtime import RuntimeLoader, VisionRuntimeError
from vision_worker import VisionWorker


class RuntimeTests(unittest.TestCase):
    def test_first_import_error_is_preserved_without_reimport(self):
        loader = RuntimeLoader()
        with patch("vision_runtime.importlib.import_module", side_effect=ImportError("causa original")) as importer:
            for _ in range(2):
                with self.assertRaisesRegex(VisionRuntimeError, "causa original"):
                    loader.load()
        self.assertEqual(importer.call_count, 1)

    def test_successful_import_is_shared_with_worker(self):
        loader = RuntimeLoader()
        module = SimpleNamespace(solutions=SimpleNamespace(face_mesh=SimpleNamespace(FaceMesh=lambda: None)))
        with patch("vision_runtime.importlib.import_module", return_value=module) as importer:
            self.assertIs(loader.load(), module)
            self.assertIs(loader.load(), module)
        self.assertEqual(importer.call_count, 1)

    def test_failed_startup_never_opens_camera(self):
        import main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"failed.json"
            with patch("sys.argv", ["main.py", "--headless", "--duration", "1", "--report", str(output)]), \
                 patch("detector.EyeDetector.preflight", side_effect=VisionRuntimeError("falha original")), \
                 patch("camera_stream.CameraCaptureWorker.start") as start:
                self.assertEqual(main.main(), 2)
            self.assertFalse(start.called)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["camera_opened"])
            self.assertEqual(report["processing_fps_observed"], 0)

    def test_failed_detector_cancels_calibration_and_stale_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            session = InteractionSession(Settings(), Path(directory)/"profile.json")
            session.calibrate()
            class Camera:
                sequence = 0
                def wait_latest(self, after):
                    time.sleep(0.01)
                    self.sequence += 1
                    return SimpleNamespace(sequence=self.sequence, timestamp=time.perf_counter(),
                                           demo=False, image=np.zeros((240, 320, 3), np.uint8))
            worker = VisionWorker(Camera(), Settings(), session=session)

            now = time.perf_counter()
            worker._times.extend((now-0.03, now))
            with patch("vision_worker.EyeDetector", side_effect=VisionRuntimeError("erro original")) as constructor:
                worker.start()
                deadline = time.perf_counter()+1
                while not worker.error_message and time.perf_counter() < deadline:
                    time.sleep(0.01)
                time.sleep(0.15)
                worker.close()
            self.assertEqual(constructor.call_count, 1)
            self.assertEqual(worker.fps, 0)
            self.assertIn("erro original", worker.error_message)
            self.assertIsNone(session.snapshot().calibration)
            self.assertFalse(session.snapshot().engine.tracking)


if __name__ == "__main__":
    unittest.main()
