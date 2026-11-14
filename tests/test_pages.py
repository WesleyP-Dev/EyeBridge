from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from core import Calibration,InteractionEngine,Observation
from config import Settings
from interaction import InteractionSession
from horizontal_calibration import HorizontalCalibrator
from hud import HUD


def obs(t,x,y=0):return Observation(t,True,.5-(x-.5)*.3,y,.28,.28)


class PageTests(unittest.TestCase):
    def setUp(self):self.engine=InteractionEngine(calibration=Calibration.paged_demo());self.now=0
    def frames(self,seconds,x,y=0):
        for _ in range(round(seconds/.05)):
            self.now+=.05;self.engine.update(obs(self.now,x,y))

    def test_all_items_are_selected_by_horizontal_gaze_and_page(self):
        for col in range(4):
            self.frames(1,.2+col*.2)
            self.assertEqual(self.engine.menu.selected,col)
        self.frames(2.4,.96)
        self.assertEqual(self.engine.page,1)
        for col in range(4):
            self.frames(1,.2+col*.2)
            self.assertEqual(self.engine.menu.selected,col+4)

    def test_navigation_does_not_confirm_and_requires_return_before_repeat(self):
        self.frames(10,.96)
        self.assertEqual(self.engine.page,1)
        self.assertFalse(self.engine.history.entries)
        self.frames(2.5,.04)
        self.assertEqual(self.engine.page,0)

    def test_vertical_noise_does_not_change_item_or_prevent_confirmation(self):
        for i in range(130):
            self.now+=.05;self.engine.update(obs(self.now,.4,(-1 if i%2 else 1)*10))
        self.assertEqual(self.engine.last_confirmation.label,'Comida')
        self.assertEqual(self.engine.y,.5)

    def test_lost_face_cancels_navigation(self):
        self.frames(1,.96)
        self.now+=.05;self.engine.update(Observation(self.now))
        self.frames(.7,.96)
        self.assertEqual(self.engine.page,0)

    def test_real_calibration_uses_twelve_horizontal_targets_with_constant_vertical(self):
        with tempfile.TemporaryDirectory() as d:
            now=[0.]
            s=InteractionSession(Settings(),Path(d)/'profile.json',clock=lambda:now[0]);s.calibrate()
            self.assertIsInstance(s.calibrator,HorizontalCalibrator)
            for x,y in s.calibrator.positions:
                index=s.calibrator.index
                while s.calibrator and s.calibrator.index==index:
                    now[0]+=.05
                    o=obs(now[0],x)
                    s.consume(SimpleNamespace(demo=False),o)
            self.assertIsNotNone(s.engine.calibration.mapping['horizontal'])
            s.engine.calibration.validate()
            h=HUD();h.render(None,s.engine,'USB','OK')
            self.assertTrue(h.paged_layout)
            self.assertEqual(h.card_rects[0],h.card_rects[4])

    def test_horizontal_drift_is_rejected_on_revisit(self):
        c=HorizontalCalibrator(Settings());t=0;previous=Calibration()
        with self.assertRaises(ValueError):
            for j,(x,y) in enumerate(c.positions):
                index=c.index
                while c.index==index:
                    t+=.05;c.update(obs(t,x+(.3 if j>=6 else 0)),previous)


if __name__=='__main__':unittest.main()
