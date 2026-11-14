from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from core import InteractionEngine, Calibration, LABELS, MenuSelector, block_center
from config import Settings
from display import drawing_size, viewport_size
from hud import HUD
from interaction import InteractionSession
from test_core import sample


class LayoutTests(unittest.TestCase):
    def test_cards_are_visible_and_mouse_matches_gaze_for_multiple_aspects(self):
        for size in ((1920,1080),(1920,1200),(1024,768),(800,480),(1280,1024)):
            hud=HUD(*size)
            engine=InteractionEngine()
            canvas=hud.render(None,engine,'USB','OK')
            self.assertEqual(canvas.shape,(hud.height,hud.width,3))
            self.assertLessEqual(hud.width*hud.height,3686400)
            self.assertAlmostEqual(hud.width/hud.height,size[0]/size[1],places=2)
            for i,(left,top,right,bottom) in enumerate(hud.card_rects):
                self.assertTrue(0 <= left < right < hud.width)
                self.assertTrue(0 <= top < bottom < hud.height)
                self.assertEqual(MenuSelector.hit(*hud.demo_position((left+right)/2,(top+bottom)/2)),i)
            left,top,right,bottom=hud.preview_rect
            self.assertLess((right-left)*(bottom-top),hud.width*hud.height*.05)
            self.assertTrue(hud.clear_hit((hud.clear_rect[0]+hud.clear_rect[2])/2,
                                          (hud.clear_rect[1]+hud.clear_rect[3])/2))

    def test_resize_invalidates_preview_and_panel(self):
        hud=HUD()
        image=np.full((480,640,3),90,np.uint8)
        engine=InteractionEngine()
        hud.render(image,engine,'USB','OK')
        previous=hud._preview
        self.assertTrue(hud.resize(1024,768))
        result=hud.render(image,engine,'USB','OK')
        self.assertEqual(result.shape,(768,1024,3))
        self.assertIsNot(previous,hud._preview)
        self.assertTrue((image==90).all())

    def test_native_fullhd_and_native_1440p_with_budgeted_4k(self):
        self.assertEqual(drawing_size(1920,1080,2073600),(1920,1080))
        self.assertEqual(drawing_size(2560,1440),(2560,1440))
        for w,h in ((3840,2160),(3440,1440),(1920,1200),(1080,1920)):
            cw,ch=drawing_size(w,h,2073600)
            self.assertLessEqual(cw*ch,2073600)
            self.assertAlmostEqual(cw/ch,w/h,places=2)
        hud=HUD(1920,1080,max_pixels=2073600)
        self.assertEqual(hud.render(None,InteractionEngine(),'USB','OK').shape,(1080,1920,3))

    def test_unsupported_viewport_query_has_fallback(self):
        class CV:
            error=RuntimeError
            def getWindowImageRect(self, name): raise self.error('Wayland')
        self.assertIsNone(viewport_size(CV(),'EyeBridge'))
        self.assertEqual(drawing_size(0,0),(1280,720))


class RealSessionTests(unittest.TestCase):
    def test_uncalibrated_camera_never_confirms_and_calibrated_camera_does(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'grid.json'
            now=[0]
            session=InteractionSession(Settings(calibration_passes=1,paged_navigation=False),path,clock=lambda:now[0])
            frame=SimpleNamespace(demo=False)
            for i in range(400):
                now[0]=i*.02
                session.consume(frame,sample(now[0]))
            self.assertFalse(session.engine.history.entries)
            session.engine.calibration=Calibration.demo()
            session.engine.reset()
            for i in range(400,670):
                now[0]=i*.02
                session.consume(frame,sample(now[0]))
            self.assertEqual(session.engine.last_confirmation.label,'Água')

    def test_calibration_saves_nine_targets_and_activates_two_dimensional_selection(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'grid.json'
            now=[0]
            session=InteractionSession(Settings(calibration_passes=1,paged_navigation=False),path,clock=lambda:now[0])
            session.calibrate()
            frame=SimpleNamespace(demo=False)
            for x,y in session.calibrator.positions:
                initial=session.calibrator.index
                while session.calibrator and session.calibrator.index==initial:
                    from core import Observation
                    now[0]+=.02
                    session.consume(frame,Observation(now[0],True,.5-.3*(x-.5),.5+.12*(y-.5)))
            self.assertTrue(path.exists())
            self.assertIsNotNone(Calibration.load(path).mapping)
            for _ in range(300):
                now[0]+=.02
                session.consume(frame,sample(now[0],7))
            self.assertEqual(session.engine.last_confirmation.label,'Conversar')
            import json
            report=json.loads(path.with_name('grid-diagnostic.json').read_text(encoding='utf-8'))
            self.assertTrue(report['accepted'])
            self.assertEqual(len(report['targets']),9)

    def test_failed_calibration_diagnostic_preserves_existing_good_profile(self):
        with tempfile.TemporaryDirectory() as d:
            import json
            from core import Observation
            path=Path(d)/'grid.json'
            Calibration.demo().save(path)
            original=path.read_bytes()
            now=[0]
            session=InteractionSession(Settings(calibration_passes=1,paged_navigation=False),path,clock=lambda:now[0])
            session.calibrate()
            frame=SimpleNamespace(demo=False)
            for x,y in session.calibrator.positions:
                initial=session.calibrator.index
                while session.calibrator and session.calibrator.index==initial:
                    now[0]+=.02
                    session.consume(frame,Observation(now[0],True,.5-.3*(x-.5),.5))
            self.assertEqual(path.read_bytes(),original)
            report=json.loads(path.with_name('grid-diagnostic.json').read_text(encoding='utf-8'))
            self.assertFalse(report['accepted'])
            self.assertIn('Movimento insuficiente',report['error'])
            self.assertEqual(len(report['targets']),9)


if __name__=='__main__': unittest.main()
