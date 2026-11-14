"""Regressões de precisão temporal e dos limites da interface horizontal."""
from dataclasses import replace
import unittest
import numpy as np

from config import Settings
from core import Calibration, InteractionEngine, Observation, OneEuro
from horizontal_calibration import HorizontalCalibrator, card_at, position, target_at
from hud import HUD


def observation(t, x):
    return Observation(t, True, .5-(x-.5)*.3, .5, .28, .28)


class PrecisionTests(unittest.TestCase):
    def engine(self):
        return InteractionEngine(calibration=Calibration.paged_demo())

    def feed(self, engine, start, seconds, x, fps=20):
        for i in range(round(seconds*fps)):
            engine.update(observation(start+(i+1)/fps, x))
        return start+round(seconds*fps)/fps

    def test_drawing_and_selection_share_gutters_at_multiple_sizes(self):
        for size in ((1280,720),(1920,1080),(1024,768)):
            h=HUD(*size);h.paged_layout=True;h.resize(*size,force=True)
            gx,_,gr,_=h.grid_rect
            for col,(left,_,right,_) in enumerate(h.card_rects[:4]):
                self.assertEqual(card_at(((left+right)/2-gx)/(gr-gx)),col)
            for x in (.1,.3,.5,.7,.9):
                self.assertIsNone(card_at(x))

    def test_staring_at_a_gutter_never_confirms(self):
        for x in (.1,.3,.5,.7,.9):
            e=self.engine();self.feed(e,0,7,x)
            self.assertIsNone(e.menu.selected)
            self.assertFalse(e.history.entries)

    def test_leaving_just_before_five_seconds_cannot_confirm_old_card(self):
        e=self.engine();t=self.feed(e,0,5.05,.2)
        self.assertGreater(e.fixation_elapsed,4.8)
        self.assertFalse(e.history.entries)
        self.feed(e,t,.1,.8)
        self.assertFalse(e.history.entries)
        self.assertIsNone(e.previous_valid_target)

    def test_visual_filter_lag_cannot_finish_navigation_after_looking_away(self):
        e=self.engine();t=0
        while e.nav_elapsed<1.4-1e-8:
            t=self.feed(e,t,.05,.96)
            self.assertLess(t,3)
        self.feed(e,t,.15,.8)
        self.assertEqual(e.page,0)

    def test_blink_discards_old_samples_without_counting_closed_time(self):
        e=self.engine();t=self.feed(e,0,3,.2);elapsed=e.fixation_elapsed
        e.update(replace(observation(t+.05,.2),gaze_valid=False))
        self.assertEqual(len(e.raw_window),0)
        e.update(observation(t+.1,.8))
        self.assertEqual(len(e.raw_window),1)
        self.assertAlmostEqual(e.fixation_elapsed,elapsed)
        self.assertIsNone(e.previous_valid_target)

    def test_window_has_similar_duration_at_ten_and_twenty_fps(self):
        for fps,count in ((10,3),(20,5)):
            e=self.engine();self.feed(e,0,2,.4,fps)
            self.assertEqual(len(e.raw_window),count)
            self.assertAlmostEqual(e.sample_times[-1]-e.sample_times[0],.2)

    def test_noisy_gaze_still_confirms_each_column_at_ten_fps(self):
        for col in range(4):
            e=self.engine()
            for i in range(75):
                x=.2+.2*col+.015*np.sin(i*1.7)
                e.update(observation((i+1)/10,x))
            self.assertEqual(e.last_confirmation.block,col)
            self.assertEqual(len(e.history.entries),1)

    def test_isolated_jump_does_not_move_to_another_card_or_count(self):
        e=self.engine();t=self.feed(e,0,3,.4);before=e.fixation_elapsed
        e.update(observation(t+.05,.96))
        self.assertEqual(e.menu.selected,1)
        self.assertAlmostEqual(e.fixation_elapsed,before)
        self.assertLess(abs(e.x-.4),.02)

    def test_adaptive_filter_reduces_steady_noise_and_step_lag(self):
        def run(cutoff,beta):
            f=OneEuro(cutoff,beta);values=[]
            for i in range(200):
                values.append(f.update(.4+.02*np.sin(i*1.7),i*.05))
            for i in range(1,61):
                if abs(f.update(.8,(199+i)*.05)-.8)<.02:
                    return np.std(values[100:]),i*.05
            self.fail('Filtro não convergiu')
        old=run(.9,.25);new=run(Settings().filter_cutoff,Settings().filter_beta)
        self.assertLess(new[0],old[0]);self.assertLess(new[1],old[1])

    def test_calibration_rejects_nearest_anchor_success_outside_drawn_target(self):
        c=HorizontalCalibrator(Settings());c.ears=[.28]*12


        normal=[.5-(x-.5)*.3 for x in c.xs]
        for target in c.ids:
            values=np.full(32,normal[target])
            if target==0:
                values[:16]-=.018;values[16:]+=.018
            c.groups.append(values)
        with self.assertRaisesRegex(ValueError,'fora da área'):
            c.finish()
        self.assertTrue(all(x>=.8 for x in c.diagnostic['validation_rates']))
        self.assertFalse(c.diagnostic['accepted'])

    def test_non_finite_eyelid_reading_is_not_collected(self):
        c=HorizontalCalibrator(Settings());previous=Calibration()
        c.update(replace(observation(1,.2),left_ear=float('nan')),previous)
        self.assertIsNone(c.since);self.assertFalse(c.samples)

    def test_non_finite_timestamp_does_not_poison_next_collection(self):
        c=HorizontalCalibrator(Settings());previous=Calibration()
        c.update(observation(float('nan'),.2),previous)
        c.update(observation(1,.2),previous)
        self.assertEqual(c.since,1)
        self.assertEqual(c.last_time,1)

    def test_monotone_mapping_works_in_both_camera_directions(self):
        for anchors in ([.8,.7,.6,.5,.4,.3],[.2,.3,.4,.5,.6,.7]):
            model={'anchors':anchors,'positions':[.04,.2,.4,.6,.8,.96]}
            for i,a in enumerate(anchors):
                self.assertEqual(target_at(position(a,model)),i)


if __name__=='__main__':unittest.main()
