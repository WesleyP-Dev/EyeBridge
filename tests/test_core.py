import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from config import Settings
from core import Calibration, Calibrator, InteractionEngine, Observation, MenuSelector, LABELS, block_center, ConfirmationHistory, OneEuro


def sample(t, index=0, ear=.28, valid=True, gaze_valid=True):
    x,y = block_center(index)
    return Observation(t, valid, .5-(x-.5)*.3, .5+(y-.5)*.12, ear, ear, gaze_valid=gaze_valid)


class FixationTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.now = InteractionEngine(), 0

    def frames(self, duration, index=0, **kwargs):
        for _ in range(round(duration/.02)):
            self.now = round(self.now+.02, 6)
            self.engine.update(sample(self.now, index, **kwargs))

    def test_selects_both_rows_and_all_eight_items(self):
        for i in range(8):
            self.frames(1, i)
            self.assertEqual(self.engine.menu.selected, i)

    def test_confirms_after_five_seconds_and_never_repeats_while_staring(self):
        self.frames(4.98)
        self.assertEqual(len(self.engine.history.entries), 0)
        self.frames(.08)
        self.assertEqual(self.engine.last_confirmation.label, 'Água')
        self.frames(20)
        self.assertEqual(len(self.engine.history.entries), 1)

    def test_must_restart_count_on_other_item(self):
        self.frames(4)
        self.frames(1, 4)
        self.assertEqual(len(self.engine.history.entries), 0)
        self.frames(4.7, 4)
        self.assertEqual(self.engine.last_confirmation.label, 'Ajuda')

    def test_short_blink_pauses_instead_of_confirming_or_counting_closed_time(self):
        self.frames(3)
        before = self.engine.fixation_elapsed
        self.frames(.2, ear=.03)
        self.assertAlmostEqual(self.engine.fixation_elapsed, before)
        self.frames(1.9)
        self.assertIsNone(self.engine.last_confirmation)
        self.frames(.3)
        self.assertEqual(len(self.engine.history.entries), 1)

    def test_long_closure_resets_and_cannot_confirm_on_reopen(self):
        self.frames(4.7)
        self.frames(.8, ear=.03)
        self.frames(.6)
        self.assertIsNone(self.engine.last_confirmation)
        self.assertLess(self.engine.fixation_elapsed, 1)

    def test_lost_face_cancels_and_history_survives(self):
        self.frames(5.1)
        self.frames(.1, valid=False)
        self.assertIsNone(self.engine.menu.selected)
        self.assertEqual(len(self.engine.history.entries), 1)
        self.frames(8)
        self.assertEqual(len(self.engine.history.entries), 1)

    def test_gap_and_duplicate_timestamps_do_not_count(self):
        self.frames(4)
        for _ in range(30):
            self.engine.update(sample(self.now))
        self.assertLess(self.engine.fixation_elapsed, 4.01)
        self.now += 4
        self.frames(.8)
        self.assertIsNone(self.engine.last_confirmation)

    def test_disagreeing_eyes_pause_then_cancel(self):
        self.frames(4.7)
        self.frames(.6, gaze_valid=False)
        self.frames(.5)
        self.assertIsNone(self.engine.last_confirmation)

    def test_same_item_can_be_requested_again_after_deliberately_looking_away(self):
        self.frames(5.1)
        self.frames(1.5, 1)
        self.frames(5.8)
        self.assertEqual([e.label for e in self.engine.history.entries], ['Água', 'Água'])

    def test_non_finite_observation_cancels(self):
        self.frames(4.8)
        self.engine.update(Observation(self.now+.02, True, math.nan))
        self.assertFalse(self.engine.tracking)
        self.assertIsNone(self.engine.menu.selected)

    def test_calibration_cannot_confirm(self):
        for i in range(400):
            self.engine.update(sample(i*.02), paused=True)
        self.assertIsNone(self.engine.last_confirmation)

    def test_neutral_header_or_gutter_cannot_select(self):
        self.assertIsNone(MenuSelector.hit(.5, .5))
        self.assertIsNone(MenuSelector.hit(.125, -.2))
        self.assertIsNone(MenuSelector.hit(1.1, .25))
        self.frames(4)
        for _ in range(60):
            self.now += .02
            self.engine.update(Observation(self.now, True, .6125, .35))
        self.assertIsNone(self.engine.menu.selected)
        self.assertIsNone(self.engine.last_confirmation)

    def test_one_closed_eye_does_not_advance_fixation(self):
        self.frames(4.8)
        for _ in range(40):
            self.now += .02
            obs=sample(self.now)
            from dataclasses import replace
            self.engine.update(replace(obs,left_ear=.03))
        self.assertIsNone(self.engine.last_confirmation)


class CalibrationTests(unittest.TestCase):
    def collect(self, transform=lambda x,y:(.5-.3*(x-.5),.5+.12*(y-.5))):
        cal, old, t, result = Calibrator(), Calibration(), 0, None
        for x,y in cal.positions:
            raw_x,raw_y = transform(x,y)
            initial=cal.index
            while cal.index == initial:
                t += .02
                result=cal.update(Observation(t,True,raw_x,raw_y,.18,.18),old)
        return result

    def test_two_axis_calibration_with_cross_axis_coupling(self):
        transform=lambda x,y:(.5-.3*(x-.5)+.03*(y-.5),.5+.12*(y-.5)+.01*(x-.5))
        profile=self.collect(transform)
        self.assertTrue(profile.calibrated)
        for i in range(8):
            point=block_center(i)
            mapped=profile.map_pair(*transform(*point))
            self.assertEqual(MenuSelector.hit(*mapped),i)
            np.testing.assert_allclose(mapped,point,atol=.015)
        self.assertAlmostEqual(profile.ear_open,.18)

    def test_overlapping_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Movimento insuficiente'):
            self.collect(lambda x,y:(.5-.3*(x-.5),.5))

    def test_curved_eye_geometry_uses_quadratic_instead_of_rejecting_user(self):

        transform=lambda x,y:(.5-.3*(x-.5),.5+.12*(y-.5)+.6*(x-.5)**2)
        profile=self.collect(transform)
        self.assertEqual(profile.mapping['kind'],'quadratic')
        for i in range(8):
            self.assertEqual(MenuSelector.hit(*profile.map_pair(*transform(*block_center(i)))),i)

        for i in range(8):
            x,y=block_center(i)
            for dx,dy in ((.025,.03),(-.025,-.03)):
                self.assertEqual(MenuSelector.hit(*profile.map_pair(*transform(x+dx,y+dy))),i)

    def test_actually_identical_card_measurements_are_not_accepted(self):
        def transform(x,y):
            if (x,y)==block_center(1):
                x,y=block_center(0)
            return .5-.3*(x-.5),.5+.12*(y-.5)
        with self.assertRaisesRegex(ValueError,'Leituras não separaram'):
            self.collect(transform)

    def test_randomized_or_mislabeled_targets_rejected(self):
        positions=iter([(.5,.5),(.6,.48),(.4,.52),(.6,.52),(.4,.48),(.55,.49),(.45,.51),(.45,.49),(.55,.51)])
        with self.assertRaises(ValueError):
            self.collect(lambda x,y:next(positions))

    def test_model_roundtrip_and_legacy_profile_is_readable(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'profile.json'
            p=self.collect()
            p.save(path)
            self.assertEqual(Calibration.load(path),p)
            Calibration().save(path)
            self.assertIsNone(Calibration.load(path).mapping)

    def test_invalid_model_rejected(self):
        for model in ({}, {'mean':[.5,.5],'scale':[0,.1],'coef':[[0,0]]},
                      {'mean':[float('nan'),.5],'scale':[.1,.1],'coef':[[0,0]]*3}):
            with self.assertRaises(ValueError):
                Calibration(mapping=model).validate()

    def test_duplicate_or_invalid_calibration_samples_cannot_finish(self):
        cal=Calibrator()
        for i in range(200):
            cal.update(sample(0),Calibration())
        self.assertEqual(cal.index,0)
        for i in range(200):
            cal.update(sample(i*.02, valid=False),Calibration())
        self.assertEqual(cal.index,0)

    def test_legacy_horizontal_mapping_both_directions(self):
        for left,right in ((.65,.35),(.35,.65)):
            p=Calibration(left=left,right=right).validate()
            self.assertAlmostEqual(p.map(left),.1)
            self.assertAlmostEqual(p.map(right),.9)

    def test_filter_constant_and_step(self):
        f=OneEuro()
        for i in range(100):
            self.assertAlmostEqual(f.update(.5,i*.05),.5)
        for i in range(10):
            output=f.update(.9,5+i*.05)
        self.assertGreater(output,.88)


class SettingsHistoryTests(unittest.TestCase):
    def test_new_defaults_and_validation(self):
        self.assertEqual(Settings().confirmation_seconds,5)
        self.assertTrue(Settings().validate().fullscreen)
        for kw in ({'confirmation_seconds':0},{'fixation_grace':1},{'repeat_release':0}, {'ear_open':math.nan}):
            with self.assertRaises(ValueError):
                Settings(**kw).validate()

    def test_original_five_preserved_and_new_items_added(self):
        self.assertEqual(LABELS[:5],('Água','Comida','Banheiro','Conforto','Ajuda'))
        self.assertEqual(len(LABELS),8)

    def test_bounded_history_and_clear(self):
        h=ConfirmationHistory(2)
        for i in range(3):h.add(i,i)
        self.assertEqual([e.label for e in h.entries],['Banheiro','Comida'])
        h.clear()
        self.assertFalse(h.entries)


if __name__=='__main__': unittest.main()
