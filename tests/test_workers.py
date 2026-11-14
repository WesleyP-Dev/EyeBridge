import time
import io
import struct
import unittest
from unittest.mock import patch
import numpy as np
from camera_stream import CameraCaptureWorker, PicameraBridge
from config import Settings
from detector import DemoInput, EyeDetector
from core import InteractionEngine, block_center, Calibration
from vision_worker import VisionWorker
from hud import HUD


class CameraTests(unittest.TestCase):
    def test_csi_bridge_decodes_frame_and_preserves_timestamp(self):
        bridge = object.__new__(PicameraBridge)
        bridge.s = Settings(width=160, height=120)
        pixels = bytes([40, 80, 120])*(160*120)
        stream = io.BytesIO(struct.pack("!dI", 12.5, len(pixels))+pixels)
        class Process:
            stdout = stream
        bridge.process = Process()
        image = bridge.capture_array("main")
        self.assertEqual(image.shape, (120, 160, 3))
        self.assertEqual(bridge.timestamp, 12.5)
        self.assertEqual(image[0, 0].tolist(), [40, 80, 120])

    def test_csi_bridge_rejects_truncated_stream(self):
        bridge = object.__new__(PicameraBridge)
        class Process:
            stdout = io.BytesIO(b"123")
        bridge.process = Process()
        with self.assertRaises(RuntimeError):
            bridge._read_exact(12)

    def test_demo_replaces_single_slot_and_stops(self):
        worker = CameraCaptureWorker(Settings(), demo=True).start()
        try:
            time.sleep(0.15)
            first = worker.latest()
            self.assertIsNotNone(first)
            self.assertTrue(first.demo)
            self.assertEqual(first.image.shape, (480, 640, 3))
            time.sleep(0.1)
            last = worker.latest(first.sequence)
            self.assertGreater(last.sequence, first.sequence)
            self.assertIsNone(worker.latest(last.sequence))
        finally:
            worker.close()
        self.assertFalse(worker._thread.is_alive())

    def test_reconnect_after_read_failure(self):
        class FakeCamera:
            released = False
            def read(self):
                return False, None
            def release(self):
                self.released = True
        fake = FakeCamera()
        worker = CameraCaptureWorker(Settings())
        with patch.object(worker, "_open", return_value=(fake, False)) as opening:
            worker.start()
            time.sleep(1.12)
            worker.close()
        self.assertGreaterEqual(opening.call_count, 2)
        self.assertTrue(fake.released)


class DetectorTests(unittest.TestCase):
    def test_camera_vision_and_engine_select_all_eight_items(self):
        settings = Settings()
        camera = CameraCaptureWorker(settings, demo=True).start()
        vision = VisionWorker(camera, settings).start()
        engine = InteractionEngine(settings)
        sequence = -1
        try:
            for region in range(8):
                vision.move_demo(*block_center(region))
                deadline = time.monotonic()+2
                selected = None
                while time.monotonic() < deadline:
                    result = vision.latest()
                    if result and result[0].sequence != sequence:
                        sequence = result[0].sequence
                        selected = engine.update(result[1])
                        if selected == region:
                            break
                    time.sleep(0.01)
                self.assertEqual(selected, region)
        finally:
            vision.close()
            camera.close()

    def test_ear_geometry_is_rotation_invariant(self):
        points = np.array([(0, 0), (2, -2), (8, -2), (10, 0), (8, 2), (2, 2)], dtype=float)
        theta = 0.7
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.assertAlmostEqual(EyeDetector._ear(points), 0.4)
        self.assertAlmostEqual(EyeDetector._ear(points @ rotation), 0.4)

    def test_demo_maps_each_region_to_open_eye_observation(self):
        for region in range(8):
            demo = DemoInput()
            demo.x, demo.y = block_center(region)
            observation = demo.observation(1)
            from core import Calibration
            self.assertTrue(observation.valid)
            self.assertAlmostEqual(Calibration.demo().map_pair(observation.gaze_x, observation.gaze_y)[0], demo.x)
            self.assertGreater(observation.left_ear, 0.2)

    def test_clear_button_clears_history_and_confirmation(self):
        engine = InteractionEngine()
        engine.last_confirmation = engine.history.add(0, 1)
        hud = HUD()
        self.assertFalse(hud.clear_history_at(engine, 0, 0))
        self.assertEqual(len(engine.history.entries), 1)
        left, top, right, bottom = hud.clear_rect
        self.assertTrue(hud.clear_history_at(engine, (left+right)//2, (top+bottom)//2))
        self.assertEqual(len(engine.history.entries), 0)
        self.assertIsNone(engine.last_confirmation)


if __name__ == "__main__":
    unittest.main()
