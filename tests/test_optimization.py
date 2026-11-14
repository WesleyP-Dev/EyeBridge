from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from config import Settings
from core import Observation, InteractionEngine
from interaction import InteractionSession
from hud import HUD
from camera_stream import CameraCaptureWorker
from metrics import StageMetrics
from camera_diagnostics import inspect_camera


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)/"profile.json"
        self.now = 0
        self.session = InteractionSession(Settings(), self.path, demo=True, clock=lambda: self.now)
        self.frame = SimpleNamespace(demo=True)

    def tearDown(self):
        self.directory.cleanup()

    def frames(self, duration, ear=0.28, x=0.6125, y=0.47):
        for _ in range(round(duration*30)):
            self.now += 1/30
            self.session.consume(self.frame, Observation(self.now, True, x, y, left_ear=ear, right_ear=ear))

    def test_confirmation_without_any_gui_poll(self):

        self.frames(5.2)
        view = self.session.snapshot()
        self.assertEqual([entry.label for entry in view.engine.history.entries], ["Água"])

    def test_snapshot_is_detached_from_mutation(self):
        self.frames(0.9)
        old = self.session.snapshot()
        self.frames(0.8, x=0.4625)
        self.assertEqual(old.engine.menu.selected, 0)
        self.assertEqual(self.session.snapshot().engine.menu.selected, 2)

    def test_idle_camera_cancels_pending_fixation(self):
        self.frames(0.9)
        self.frames(0.5, ear=0.06)
        self.now += 0.4
        self.session.idle()
        self.frames(0.8)
        self.assertEqual(len(self.session.snapshot().engine.history.entries), 0)

    def test_clear_and_mode_switch_preserve_safety(self):
        self.frames(5.2)
        self.session.clear()
        self.assertEqual(len(self.session.snapshot().engine.history.entries), 0)
        self.session.set_demo(False)
        self.frames(1)
        self.assertFalse(self.session.snapshot().engine.tracking)

    def test_demo_does_not_write_real_calibration(self):
        self.session.gain(True)
        self.session.calibrate()
        self.assertFalse(self.path.exists())
        self.assertIsNone(self.session.snapshot().calibration)


class CacheTests(unittest.TestCase):
    def test_moving_reticle_reuses_panel_and_updates_pixels(self):
        hud = HUD()
        engine = InteractionEngine()
        engine.tracking = True
        first = hud.render(None, engine, "USB", "OK", now=1)
        engine.x = 0.2
        second = hud.render(None, engine, "USB", "OK", now=1.1)
        self.assertEqual(hud.panel_builds, 1)
        self.assertFalse(np.array_equal(first, second))

    def test_history_clear_invalidates_cached_panel(self):
        hud, engine = HUD(), InteractionEngine()
        engine.history.add(0, 1)
        first = hud.render(None, engine, "USB", "OK", now=1)
        engine.history.clear()
        second = hud.render(None, engine, "USB", "OK", now=1.1)
        self.assertEqual(hud.panel_builds, 2)
        self.assertFalse(np.array_equal(first, second))

    def test_preview_cache_does_not_modify_capture(self):
        hud, engine = HUD(), InteractionEngine()
        image = np.full((480, 640, 3), 99, np.uint8)
        hud.render(image, engine, "USB", "OK", now=1)
        cached = hud._preview
        hud.render(image, engine, "USB", "OK", now=1.1)
        self.assertIs(cached, hud._preview)
        self.assertTrue((image == 99).all())


class DiagnosticsTests(unittest.TestCase):
    def test_samples_are_bounded(self):
        metrics = StageMetrics(3)
        for i in range(10):
            metrics.record(model_ms=i)
        self.assertEqual(metrics.summary()["model_ms"]["samples"], 3)
        self.assertEqual(metrics.summary()["model_ms"]["median"], 8)

    def test_diagnostics_never_sets_camera_controls(self):
        completed = SimpleNamespace(returncode=0, stdout="ok", stderr="")
        with patch("camera_diagnostics.platform.system", return_value="Linux"), \
             patch("camera_diagnostics.shutil.which", return_value="v4l2-ctl"), \
             patch("camera_diagnostics.subprocess.run", return_value=completed) as run:
            result = inspect_camera(2)
        self.assertEqual(len(result["commands"]), 4)
        for call in run.call_args_list:
            self.assertFalse(any(arg.startswith("--set") for arg in call.args[0]))

    def test_mjpeg_failure_falls_back_without_changing_requested_resolution(self):
        worker = CameraCaptureWorker(Settings())
        class Camera:
            def read(self):
                return False, None
            def release(self):
                pass
        opened = []
        def opening():
            opened.append((worker._native_usb, worker.s.width, worker.s.height))
            if len(opened) == 2:
                worker._stop.set()
            return Camera(), False

        with patch.object(worker, "_open", side_effect=opening), \
             patch.object(worker._stop, "wait", return_value=False):
            worker._run()
        self.assertEqual(opened, [(False, 640, 480), (True, 640, 480)])


if __name__ == "__main__":
    unittest.main()
