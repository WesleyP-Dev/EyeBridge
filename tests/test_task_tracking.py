from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
import numpy as np
from core import Calibration,Calibrator,Observation,MenuSelector,block_center,InteractionEngine
from detector import EyeDetector


def reading(t,x,y):

    rawx=.5-.3*(x-.5)
    return Observation(t,True,rawx,.446,.28,.28,features=(rawx,y-.5,.4,.28,0.,0.))


class TaskTrackingTests(unittest.TestCase):
    def fit(self):
        c,t,result=Calibrator(),0,None
        for x,y in c.positions:
            index=c.index
            for _ in range(120):
                t+=.02
                result=c.update(reading(t,x,y),Calibration())
                if c.index!=index:break
            else:raise AssertionError('Calibração não avançou')
        return result,c

    def test_learned_signal_separates_rows_when_all_old_vertical_signals_constant(self):
        p,c=self.fit()
        self.assertEqual(p.mapping['signal'],'cards-v2')
        self.assertEqual(len(p.mapping['mean']),6)
        self.assertEqual(len(c.diagnostic['features']),6)
        for i in range(8):
            x,y=block_center(i)
            obs=reading(0,x,y)
            self.assertEqual(MenuSelector.hit(*p.map_pair(obs.gaze_x,obs.gaze_y,obs.features)),i)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'profile.json';p.save(path)
            self.assertEqual(Calibration.load(path),p)

    def test_six_signal_engine_confirms_at_ten_fps_and_cancels_missing_signals(self):
        p,_=self.fit();e=InteractionEngine(calibration=p)
        x,y=block_center(7)
        for i in range(60):e.update(reading(i*.1,x,y))
        self.assertEqual(e.last_confirmation.label,'Conversar')
        e.update(Observation(6.1,True))
        self.assertFalse(e.tracking)

    def test_blendshapes_and_pose_are_extracted_and_blinks_rejected(self):
        names=('eyeLookDownLeft','eyeLookDownRight','eyeLookUpLeft','eyeLookUpRight','eyeBlinkLeft','eyeBlinkRight')
        values=[.7,.6,.1,.2,0,0]
        cats=[NS(category_name=n,score=v) for n,v in zip(names,values)]
        result=NS(face_blendshapes=[cats],facial_transformation_matrixes=[np.eye(4)])
        features,blink=EyeDetector.task_features(result,.5,.4,.28)
        self.assertAlmostEqual(features[1],.5)
        self.assertEqual(features[4:],(0,0))
        self.assertFalse(blink)
        cats[-1].score=.9
        self.assertTrue(EyeDetector.task_features(result,.5,.4,.28)[1])
        result.face_blendshapes=[]
        self.assertIsNone(EyeDetector.task_features(result,.5,.4,.28)[0])

    def test_invalid_task_signal_cancels_selection(self):
        p,_=self.fit();e=InteractionEngine(calibration=p)
        obs=reading(0,.125,.25)
        e.update(replace(obs,features=(.5,float('nan'),.4,.28,0,0)))
        self.assertFalse(e.tracking)


if __name__=='__main__':unittest.main()
