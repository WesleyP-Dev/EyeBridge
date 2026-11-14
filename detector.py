"""Detecta os olhos com o Face Landmarker."""
import cv2
import numpy as np
import time
from pathlib import Path
from core import Observation
from vision_runtime import runtime


class EyeDetector:

    EYES = (
        ((33, 160, 158, 133, 153, 144), 468),
        ((362, 385, 387, 263, 373, 380), 473),
    )

    def __init__(self):
        mp = runtime.load()
        self.mp = mp
        model = Path(__file__).with_name('models')/'face_landmarker.task'

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=False,
            min_face_detection_confidence=.6, min_tracking_confidence=.6)
        self.mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.last_ms = -1
        self.timings = {}

    @classmethod
    def preflight(cls):
        """Valida pacote, modelos locais e inferência antes de abrir câmera/GUI."""
        detector = cls()
        try:
            detector.process(np.zeros((240, 320, 3), np.uint8), time.perf_counter())
        finally:
            detector.close()

    @staticmethod
    def _ear(points):
        a, b, c, d, e, f = points
        width = np.linalg.norm(a-d)
        if width < 5:
            return None
        return float((np.linalg.norm(b-f)+np.linalg.norm(c-e))/(2*width))

    @staticmethod
    def _eye_measure(gray, landmarks, indices, iris):
        points = landmarks[list(indices)]
        a, _, _, d, _, _ = points
        width = float(np.linalg.norm(d-a))
        ear = EyeDetector._ear(points)
        if ear is None:
            return None
        direction = (d-a)/width
        normal = np.array([-direction[1], direction[0]])
        mid = (a+d)/2
        pupil = landmarks[iris]
        ratio = float(np.dot(pupil-a, direction)/width)
        vertical = float(np.dot(pupil-mid, normal)/width)

        upper = (points[1]+points[2])/2
        lower = (points[4]+points[5])/2
        lid_height = float(np.dot(lower-upper, normal))
        between = float(np.dot(pupil-upper, normal)/lid_height) if lid_height > width*.04 else 0.5
        between = float(np.clip(between, -0.5, 1.5))

        scale = 52/width
        matrix = np.array([
            [direction[0]*scale, direction[1]*scale, 32-np.dot(mid, direction)*scale],
            [normal[0]*scale, normal[1]*scale, 32-np.dot(mid, normal)*scale],
        ], dtype=np.float32)
        patch = cv2.warpAffine(gray, matrix, (64, 64), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        contrast_ratio = ratio

        if ear > 0.12:
            px = int(np.clip(6+ratio*52, 10, 54))
            py = int(np.clip(32+vertical*52, 10, 54))
            roi = patch[py-7:py+8, px-8:px+9]

            transformed = cv2.transform(points[None], matrix)[0]
            eyelid = np.zeros((64, 64), np.uint8)
            cv2.fillConvexPoly(eyelid, np.round(transformed).astype(np.int32), 1)
            mask = eyelid[py-7:py+8, px-8:px+9].astype(np.float32)
            weights = np.maximum(0, np.percentile(roi, 45)-roi.astype(np.float32))*mask
            total = float(weights.sum())
            if total > 30:
                cx = float((weights.sum(axis=0)*np.arange(roi.shape[1])).sum()/total)
                contrast_ratio = (px-8+cx-6)/52
                contrast_ratio = float(np.clip(contrast_ratio, ratio-0.06, ratio+0.06))
        fused = 0.95*ratio+0.05*contrast_ratio
        weight = width*float(np.clip(ear, 0.10, 0.35))
        return fused, vertical, ear, tuple(float(v) for v in pupil), weight, between

    def process(self, image, timestamp):

        start = time.perf_counter()
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        model_start = time.perf_counter()
        self.last_ms = max(self.last_ms+1, int(timestamp*1000))
        result = self.mesh.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb),self.last_ms)
        model_end = time.perf_counter()
        self.timings = {"color_ms": (model_start-start)*1000,
                        "model_ms": (model_end-model_start)*1000, "eye_geometry_ms": 0.0}
        if not result.face_landmarks:
            return Observation(timestamp)
        lm = result.face_landmarks[0]
        if len(lm) < 478:
            return Observation(timestamp)
        points = np.array([(p.x*width, p.y*height) for p in lm], dtype=np.float32)
        if not np.isfinite(points).all():
            return Observation(timestamp)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes = [self._eye_measure(gray, points, *eye) for eye in self.EYES]
        if any(eye is None for eye in eyes):
            return Observation(timestamp)
        weights = [eye[4] for eye in eyes]
        x = float(np.average([eye[0] for eye in eyes], weights=weights))
        y = float(np.average([eye[1] for eye in eyes], weights=weights))
        between = float(np.average([eye[5] for eye in eyes], weights=weights))
        opening = float(np.average([eye[2] for eye in eyes], weights=weights))
        if not (0.02 < x < 0.98 and all(0 <= eye[2] < 1 for eye in eyes)):
            return Observation(timestamp)
        consistent = abs(eyes[0][0]-eyes[1][0]) <= 0.18
        features, blinking = self.task_features(result, x, between, opening)
        if features is None:
            return Observation(timestamp)
        self.timings["eye_geometry_ms"] = (time.perf_counter()-model_end)*1000
        return Observation(timestamp, True, x, float(np.clip(0.5+y, 0, 1)),
                           eyes[0][2], eyes[1][2], tuple(eye[3] for eye in eyes), consistent and not blinking,
                           features)

    @staticmethod
    def task_features(result, x, between, opening):
        if not result.face_blendshapes:
            return None, False
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        names = ('eyeLookDownLeft','eyeLookDownRight','eyeLookUpLeft','eyeLookUpRight',
                 'eyeBlinkLeft','eyeBlinkRight')
        if not all(n in scores for n in names):
            return None, False
        vertical = (scores[names[0]]+scores[names[1]]-scores[names[2]]-scores[names[3]])/2


        values = (x,float(vertical),between,opening,0.,0.)
        if not np.isfinite(values).all():
            return None, False
        return values, max(scores['eyeBlinkLeft'],scores['eyeBlinkRight']) > .65

    def close(self):
        self.mesh.close()


class DemoInput:
    """Simulação 2D, com o mesmo tempo de confirmação da câmera real."""
    def __init__(self, automatic=False):
        self.x, self.y = 0.125, 0.25
        self.automatic = automatic
        self.started = None
        self.closed_until = 0

    def blink(self, now):
        self.closed_until = now+0.15

    def observation(self, now):
        if self.started is None:
            self.started = now
        if self.automatic:
            from core import block_center, LABELS
            elapsed=(now-self.started)%60
            if 28<=elapsed<30:self.x,self.y=.96,.5
            elif elapsed>=58:self.x,self.y=.04,.5
            else:
                local=int((elapsed if elapsed<28 else elapsed-30)/7)%4
                self.x,self.y=.2+.2*local,.5
        raw_x = 0.5-(self.x-0.5)*0.3
        raw_y = 0.5+(self.y-0.5)*0.12
        ear = 0.06 if now < self.closed_until else 0.28
        return Observation(now, True, raw_x, raw_y, ear, ear)
