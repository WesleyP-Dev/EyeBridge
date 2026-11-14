"""Calibra o olhar e controla as confirmações."""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from statistics import median
from collections import deque
from datetime import datetime
from config import Settings
from horizontal_calibration import card_at, navigation_at
import numpy as np

LABELS = ("Água", "Comida", "Banheiro", "Conforto", "Ajuda", "Sono", "Dor", "Conversar")
COLUMNS, ROWS = 4, 2


def block_center(index):
    return ((index % COLUMNS + 0.5)/COLUMNS, (index//COLUMNS + 0.5)/ROWS)


def mapping_features(x, kind='affine'):
    """Base linear ou quadrática pequena; não aumenta o modelo neural."""
    x = np.atleast_2d(x)
    a, b = x[:, 0], x[:, 1]
    if kind == 'quadratic':
        return np.column_stack((np.ones(len(x)), a, b, a*a, a*b, b*b))
    return np.column_stack((np.ones(len(x)), x))


@dataclass(frozen=True)
class Observation:
    timestamp: float
    valid: bool = False
    gaze_x: float = 0.5
    gaze_y: float = 0.5
    left_ear: float = 0.28
    right_ear: float = 0.28
    pupils: tuple = ()
    gaze_valid: bool = True
    features: tuple = ()
    head_pose: tuple = ()
    face_points: tuple = ()

    def vector(self):
        return np.asarray(self.features if self.features else (self.gaze_x, self.gaze_y), dtype=float)


@dataclass(frozen=True)
class Confirmation:
    block: int
    timestamp: float
    clock: str

    @property
    def label(self):
        return LABELS[self.block]


class ConfirmationHistory:
    def __init__(self, limit=8):
        self.entries = deque(maxlen=limit)

    def add(self, block, timestamp):
        entry = Confirmation(block, timestamp, datetime.now().strftime("%H:%M:%S"))
        self.entries.appendleft(entry)
        return entry

    def clear(self):
        self.entries.clear()


@dataclass
class Calibration:
    center: float = 0.5
    left: float = 0.62
    right: float = 0.38
    ear_open: float = 0.28
    gain: float = 1.0
    calibrated: bool = False
    anchors: list | None = None
    mapping: dict | None = None

    def validate(self):
        values = (self.center, self.left, self.right, self.ear_open, self.gain)
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values):
            raise ValueError("Calibração não finita")
        if not (0 <= self.left <= 1 and 0 <= self.center <= 1 and 0 <= self.right <= 1
                and (self.left-self.center)*(self.right-self.center) < 0
                and min(abs(self.left-self.center), abs(self.right-self.center)) >= 0.012
                and 0.05 < self.ear_open < 0.8 and 0.25 <= self.gain <= 4
                and type(self.calibrated) is bool):
            raise ValueError("Calibração sem amplitude ocular suficiente ou fora dos limites")
        if self.anchors is not None:
            if len(self.anchors) != 5 or not all(isinstance(a, (int, float)) and math.isfinite(a) and 0 <= a <= 1 for a in self.anchors):
                raise ValueError("Calibração de cinco pontos inválida")
            steps = [b-a for a, b in zip(self.anchors, self.anchors[1:])]
            if not (all(s >= 0.006 for s in steps) or all(s <= -0.006 for s in steps)):
                raise ValueError("Alvos sobrepostos. Repita olhando para cada cartão.")
            if any(abs(a-b) > 1e-6 for a, b in zip((self.left, self.center, self.right), (self.anchors[0], self.anchors[2], self.anchors[4]))):
                raise ValueError("Centro e extremos não correspondem aos cinco alvos")
        if self.mapping is not None:
            try:
                mean = np.asarray(self.mapping['mean'], dtype=float)
                scale = np.asarray(self.mapping['scale'], dtype=float)
                coef = np.asarray(self.mapping['coef'], dtype=float)
                kind = self.mapping.get('kind', 'affine')
                dimensions = len(mean)
                if self.mapping.get('horizontal') is not None:
                    h=self.mapping['horizontal'];a=np.asarray(h['anchors'],float);p=np.asarray(h['positions'],float)
                    if (a.shape!=(6,) or p.shape!=(6,) or not np.isfinite(a).all() or not np.isfinite(p).all()
                            or not ((np.diff(a)>.008).all() or (np.diff(a)<-.008).all())
                            or (np.diff(p)<=0).any()):raise ValueError()
                if self.mapping.get('classifier') is not None:
                    from card_classifier import validate
                    validate(self.mapping['classifier'])
                if (kind not in ('affine', 'quadratic') or dimensions not in (2, 4, 6)
                        or mean.shape != (dimensions,) or scale.shape != (dimensions,)
                        or (kind == 'quadratic' and dimensions != 2)
                        or coef.shape != ((6 if kind == 'quadratic' else dimensions+1), 2)
                        or not all(np.isfinite(a).all() for a in (mean, scale, coef))
                        or (scale < 1e-6).any()):
                    raise ValueError()
                for key in ('error_radius','head_error'):
                    if self.mapping.get(key) is not None:
                        radius=np.asarray(self.mapping[key],float)
                        if radius.shape!=(2,) or not np.isfinite(radius).all() or (radius<0).any():raise ValueError()
                if self.mapping.get('x_anchors') is not None:
                    anchors=np.asarray(self.mapping['x_anchors'],float)
                    if (anchors.shape!=(4,) or not np.isfinite(anchors).all()
                            or not ((np.diff(anchors)>.006).all() or (np.diff(anchors)<-.006).all())):raise ValueError()
                if self.mapping.get('head_compensation') is not None:
                    h=self.mapping['head_compensation']
                    origin=np.asarray(h['origin'],float);n=len(origin)
                    arrays=[origin,np.asarray(h['coef'],float),np.asarray(h['low'],float),np.asarray(h['high'],float)]
                    if (dimensions!=6 or n not in (2,6) or arrays[1].shape!=(n,4)
                            or arrays[2].shape!=(n,) or arrays[3].shape!=(n,)
                            or not all(np.isfinite(a).all() for a in arrays)
                            or (arrays[2]>=arrays[3]).any()):raise ValueError()
            except (KeyError, TypeError, ValueError):
                raise ValueError('Modelo de calibração 2D inválido') from None
        return self

    @classmethod
    def demo(cls):
        return cls(calibrated=True, mapping={'mean': [0.5, 0.5], 'scale': [0.15, 0.06],
                   'coef': [[0.5, 0.5], [-0.5, 0], [0, 0.5]]})

    @classmethod
    def paged_demo(cls):
        result=cls.demo()
        xs=[.04,.2,.4,.6,.8,.96]
        result.mapping['horizontal']={'positions':xs,'anchors':[.5-(x-.5)*.3 for x in xs]}
        return result

    def map_pair(self, x, y, features=None, head_pose=None):
        if self.mapping is None:
            return self.map(x), 0.5+3*(y-0.5)
        m = self.mapping
        if m.get('horizontal') is not None:
            from horizontal_calibration import position
            return position(x,m['horizontal']),.5
        if m.get('classifier') is not None:
            from card_classifier import position
            return position(features,m['classifier'])
        values = np.asarray(features if len(m['mean']) > 2 else [x, y], dtype=float)
        if values.shape != (len(m['mean']),) or not np.isfinite(values).all():
            raise ValueError('Leitura ocular incompatível; refaça G com os fontes atualizados')
        z = mapping_features((values-m['mean'])/m['scale'], m.get('kind', 'affine'))
        px, py = (z @ np.asarray(m['coef']))[0]
        if m.get('x_anchors') is not None:
            a=np.asarray(m['x_anchors'])
            targets=np.array([.125,.375,.625,.875])
            if a[-1]<a[0]:a=a[::-1];targets=targets[::-1]
            index=max(0,min(2,int(np.searchsorted(a,values[0]))-1))
            px=targets[index]+(values[0]-a[index])/(a[index+1]-a[index])*(targets[index+1]-targets[index])

        return float(0.5+(px-0.5)*self.gain), float(py)

    def map(self, raw):
        if self.anchors is not None:
            anchors = self.anchors
            sign = 1 if anchors[-1] > anchors[0] else -1
            index = 0
            while index < 3 and raw*sign > anchors[index+1]*sign:
                index += 1
            fraction = (raw-anchors[index])/(anchors[index+1]-anchors[index])
            position = 0.1+0.2*(index+fraction)
            return max(0, min(1, 0.5+(position-0.5)*self.gain))

        side = self.left if (raw-self.center)*(self.left-self.center) > 0 else self.right
        delta = (-0.4 if side == self.left else 0.4) * (raw-self.center)/(side-self.center)
        return max(0.0, min(1.0, 0.5 + delta*self.gain))

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))).validate()


class Calibrator:
    """Nove alvos; testa ajuste linear e não linear com amostras reservadas."""
    def __init__(self, guided=True, settings=None):
        self.settings = settings
        self.settle = settings.calibration_settle if settings else 1
        self.collect = settings.calibration_collect if settings else 1.2
        self.window = settings.tracking_median if settings else 5

        self.order = (None, *range(len(LABELS)))
        self.revisit=bool(settings and settings.calibration_passes==2)
        if self.revisit:
            self.order=(None,0,4,1,5,2,6,3,7,None,2,6,0,4,3,7,1,5)
        self.targets=tuple('CENTRO' if i is None else LABELS[i].upper() for i in self.order)
        self.positions=tuple((.5,.5) if i is None else block_center(i) for i in self.order)
        self.index = 0
        self.since = self.last_time = None
        self.samples, self.points, self.ears = [], [], []
        self.groups = []
        self.diagnostic = {'version': 4, 'app_version':'3.7', 'targets': [], 'models': [], 'accepted': False}
        self.feature_count = None
        self.error = ""

    @property
    def target(self):
        return self.targets[self.index]

    @property
    def position(self):
        return self.positions[self.index]

    def update(self, obs, previous):
        if self.last_time is not None and obs.timestamp <= self.last_time:
            return None
        gap = self.last_time is not None and obs.timestamp-self.last_time > (self.settings.max_gap if self.settings else .25)
        self.last_time = obs.timestamp
        vector = obs.vector()
        if (gap or not obs.valid or not obs.gaze_valid or vector.shape not in ((2,), (4,), (6,))
                or not np.isfinite(vector).all()
                or not all(math.isfinite(v) for v in (obs.gaze_x, obs.gaze_y, obs.left_ear, obs.right_ear))
                or min(obs.left_ear, obs.right_ear) < max(0.06, previous.ear_open*0.4)):
            self.since = None
            self.samples.clear()
            return None
        if self.since is None:
            self.since = obs.timestamp
        if self.feature_count is None:
            self.feature_count = len(vector)
            self.diagnostic['features'] = (['iris_x','model_look_vertical','iris_between_lids','mean_ear','reserved_1','reserved_2'] if self.feature_count == 6 else ['iris_x', 'iris_y', 'iris_between_lids', 'mean_ear']
                                          if self.feature_count == 4 else ['iris_x', 'iris_y'])
        if len(vector) != self.feature_count:
            raise ValueError('As medidas oculares mudaram durante a coleta. Reinicie e refaça G.')
        elapsed = obs.timestamp-self.since
        if elapsed >= self.settle:
            self.samples.append((*vector, min(obs.left_ear, obs.right_ear)))
        if elapsed < self.settle+self.collect or len(self.samples) < 10:
            return None
        values = np.asarray(self.samples)
        spread = np.percentile(values[:, :-1], 90, axis=0)-np.percentile(values[:, :-1], 10, axis=0)
        if (spread > ([0.035, 0.025, 0.20, 0.08] if self.feature_count == 4 else ([0.035,.35,.20,.08,.12,.12] if self.feature_count == 6 else [0.035, 0.025]))).any():
            self.error = "Olhar instável. Fixe o ponto e mantenha a cabeça estável."
            self.since = None
            self.samples.clear()
            return None
        self.points.append(np.median(values[:, :-1], axis=0))
        self.ears.append(float(np.median(values[:, -1])))
        self.groups.append(values[:, :-1].copy())
        self.diagnostic['targets'].append({'name': self.target, 'position': list(self.position),
            'samples': len(values), 'median': self.points[-1].tolist(), 'spread_10_90': spread.tolist(),
            'readings':values[:,:-1].tolist()})
        self.samples.clear()
        self.since = None
        self.error = ""
        self.index += 1
        if self.index < len(self.targets):
            return None
        if self.feature_count==6:
            from card_classifier import fit
            canonical=(None,*range(8))
            if self.revisit:
                first=[self.groups[self.order[:9].index(i)] for i in canonical]
                second=[self.groups[9+self.order[9:].index(i)] for i in canonical]
                model,report=fit(first,self.window,second)
            else:model,report=fit(self.groups,self.window)
            report['protocol']='independent_revisit' if self.revisit else 'same_target_split'
            self.diagnostic['classifier_validation']=report
            if model is None:
                names=', '.join('CENTRO' if i==0 else LABELS[i-1].upper() for i in report['failed_targets'][:3])
                raise ValueError(f'Leituras dos cartões ainda parecidas: {names}. Refaça G.')
            self.diagnostic.update(accepted=True,selected_model='direct_cards')
            return Calibration(ear_open=median(self.ears),calibrated=True,mapping={
                'kind':'affine','mean':[0.]*6,'scale':[1.]*6,'coef':[[0.,0.]]*7,
                'signal':'cards-v2','classifier':model}).validate()
        X, Y = np.asarray(self.points), np.asarray(self.positions)
        mean, scale = X.mean(axis=0), X.std(axis=0)
        self.diagnostic['feature_std'] = scale.tolist()
        if scale[0] < .004 or (scale[1] < .002 if self.feature_count == 2 else max(scale[1:]) < .002):
            raise ValueError("Movimento insuficiente nos dois eixos. Refaça G olhando cada ponto.")

        scale = np.maximum(scale, [0.004,.03,.02,.01,.03,.03] if self.feature_count == 6 else ([0.004, 0.002, 0.02, 0.01] if self.feature_count == 4 else [0.004, 0.002]))
        if np.linalg.matrix_rank(mapping_features((X-mean)/scale)) < 3:
            raise ValueError("Não foi possível distinguir as duas linhas. Refaça G.")


        training = np.asarray([np.median(g[:len(g)//2], axis=0) for g in self.groups])
        noise = np.median([np.percentile(g,90,axis=0)-np.percentile(g,10,axis=0)
                           for g in self.groups],axis=0)/2.56
        for kind in (('affine',) if self.feature_count > 2 else ('affine', 'quadratic')):
            def fit(points):
                design = mapping_features((points-mean)/scale, kind)
                penalty = np.eye(design.shape[1])*0.001*len(X)
                penalty[0, 0] = 0
                if self.feature_count > 2:


                    penalty = np.diag(np.r_[0, .005+.25*(noise/scale)**2])*len(X)
                coef = np.linalg.solve(design.T @ design+penalty, design.T @ Y)
                if self.feature_count > 2:

                    columns = [0,1,6] if self.feature_count == 6 else [0,1]
                    h = design[:, columns]
                    coef[:, 0] = 0
                    hp = np.diag([0,.02,.25*len(X)] if self.feature_count == 6 else [0,.02])
                    coef[columns, 0] = np.linalg.solve(h.T @ h+hp,h.T @ Y[:, 0])
                return coef
            coef = fit(training)
            rates, bad = [], []
            for i, group in enumerate(self.groups):
                raw_reserved = group[len(group)//2:]

                reserved = np.asarray([np.median(raw_reserved[j:j+self.window], axis=0)
                                       for j in range(max(1, len(raw_reserved)-self.window+1))])
                pred = mapping_features((reserved-mean)/scale, kind) @ coef
                if i == 0:
                    passed = np.all(np.abs(pred-Y[i]) <= [.12, .22], axis=1)
                else:
                    passed = np.array([MenuSelector.hit(*p) == i-1 for p in pred])
                rate = float(passed.mean())
                rates.append(rate)
                if rate < .75:
                    bad.append(self.targets[i])

            final_coef = fit(X)
            predicted = mapping_features((X-mean)/scale, kind) @ final_coef
            centers_ok = all(MenuSelector.hit(*predicted[i]) == i-1 for i in range(1, 9))
            self.diagnostic['models'].append({'kind': kind, 'validation_rates': rates,
                'failed_targets': bad, 'predicted_centers': predicted.tolist(),
                'max_center_error': np.abs(predicted-Y).max(axis=0).tolist()})
            if not bad and centers_ok:
                self.diagnostic.update(accepted=True, selected_model=kind)
                result=Calibration(ear_open=median(self.ears), calibrated=True,
                    mapping={'kind': kind, 'mean': mean.tolist(), 'scale': scale.tolist(),
                             'coef': final_coef.tolist(),
                             'signal': 'task-v2' if self.feature_count == 6 else ('eyelid-v1' if self.feature_count == 4 else 'legacy')}).validate()
                if self.feature_count==6:
                    anchors=[float(np.median(X[[1+i,5+i],0])) for i in range(4)]
                    dif=np.diff(anchors)
                    if (dif>.006).all() or (dif<-.006).all():
                        result.mapping['x_anchors']=anchors
                    validation_mapping=dict(result.mapping,coef=coef.tolist())
                    if 'x_anchors' in validation_mapping:
                        va=[float(np.median(training[[1+i,5+i],0])) for i in range(4)]
                        steps=np.diff(va)
                        if (steps>.006).all() or (steps<-.006).all():validation_mapping['x_anchors']=va
                        else:validation_mapping.pop('x_anchors')
                    validation_profile=Calibration(mapping=validation_mapping)
                    errors=[]

                    for i,g in enumerate(self.groups):
                        raw=g[len(g)//2:]
                        for j in range(max(1,len(raw)-self.window+1)):
                            v=np.median(raw[j:j+self.window],axis=0)
                            pred=np.array(validation_profile.map_pair(v[0],v[1],v))
                            errors.append(np.abs(pred-Y[i]))
                    radius=np.maximum(.01,np.percentile(errors,90,axis=0))
                    result.mapping['error_radius']=radius.tolist()
                    self.diagnostic['estimated_error_radius']=radius.tolist()
                return result
        names = ', '.join(self.diagnostic['models'][-1]['failed_targets'][:3]) or 'alguns cartões'
        raise ValueError(f"Leituras não separaram {names}. Diagnóstico salvo; tente G novamente.")


class OneEuro:
    def __init__(self, cutoff=1.4, beta=1.2):
        self.cutoff, self.beta = cutoff, beta
        self.reset()

    def reset(self):
        self.x = self.t = None
        self.dx = 0.0
        self.raw = None

    @staticmethod
    def alpha(cutoff, dt):
        return 1/(1+1/(2*math.pi*cutoff*dt))

    def update(self, x, t):
        if self.t is None:
            self.x, self.t = x, t
            self.raw = x
            return x
        dt = max(0.001, min(t-self.t, 0.5))
        speed = (x-self.raw)/dt
        self.dx += self.alpha(1, dt)*(speed-self.dx)
        self.x += self.alpha(self.cutoff+self.beta*abs(self.dx), dt)*(x-self.x)
        self.t = t
        self.raw = x
        return self.x


class MenuSelector:
    """Seleciona somente cartões reais, com pequenos espaços neutros entre eles."""
    def __init__(self, settings):
        self.s = settings
        self.selected = None
        self.cancel()

    def cancel(self, clear=False):
        self.candidate = self.since = None
        self.progress = 0.0
        if clear:
            self.selected = None

    @staticmethod
    def hit(x, y):
        if not (0 <= x < 1 and 0 <= y < 1):
            return None
        col, row = int(x*4), int(y*2)

        if not (col/4+0.009 <= x <= (col+1)/4-0.009
                and row/2+0.018 <= y <= (row+1)/2-0.018):
            return None
        return row*4+col

    def update(self, x, now, y=0.25):
        target = self.hit(x, y)
        return self.update_target(target,now)

    def update_target(self,target,now):
        if target is None:
            self.cancel(clear=True)
            return None
        if target != self.candidate:
            self.candidate, self.since = target, now
        self.progress = min(1, (now-self.since)/self.s.dwell)
        self.selected = target if self.progress >= 1 else None
        return self.selected


class InteractionEngine:
    """Confirma por 5 s de olhar válido; piscadas apenas pausam, nunca confirmam."""
    def __init__(self, settings=None, calibration=None):
        self.s = settings or Settings()
        self.calibration = calibration or Calibration.demo()
        self.menu = MenuSelector(self.s)
        self.filter = OneEuro(self.s.filter_cutoff,self.s.filter_beta)
        self.vertical_filter = OneEuro(self.s.filter_cutoff,self.s.filter_beta)
        self.x = self.y = 0.5
        self.history = ConfirmationHistory(self.s.history_limit)
        self.last_confirmation = None
        self.blocked = None
        self.page=0
        self.release_elapsed = 0.0
        self.raw_window, self.vertical_window = deque(maxlen=self.s.tracking_median), deque(maxlen=self.s.tracking_median)
        self.feature_window = deque(maxlen=self.s.tracking_median)
        self.sample_times = deque(maxlen=self.s.tracking_median)
        self.reset()

    def reset(self):
        self.menu.cancel(clear=True)
        self.filter.reset()
        self.vertical_filter.reset()
        self.last_time = self.last_valid_time = None
        self.stable_target = self.pending_target = self.pending_since = None
        self.last_supported = None
        self.nav_side=None;self.nav_elapsed=0.;self.nav_progress=0.;self.nav_blocked=None
        self.previous_nav_supported=False
        self.error_margin=(0.,0.)
        self.quality_message=''
        self.face_points=()
        self.eyes_open = self.tracking = False
        self.fixation = None
        self.fixation_elapsed = self.confirm_progress = 0.0
        self.previous_valid_target = None
        self.release_elapsed = 0.0
        self.raw_window.clear()
        self.vertical_window.clear()
        self.feature_window.clear()
        self.sample_times.clear()

    def _cancel_fixation(self):
        self.fixation = None
        self.fixation_elapsed = self.confirm_progress = 0.0
        self.previous_valid_target = None

    def update(self, obs, paused=False):
        values = (obs.timestamp, obs.gaze_x, obs.gaze_y, obs.left_ear, obs.right_ear)
        if not all(math.isfinite(v) for v in values) or not obs.valid:
            self.reset()
            return None
        if (self.calibration.mapping is not None and len(self.calibration.mapping['mean']) > 2
                and (obs.vector().shape != (len(self.calibration.mapping['mean']),) or not np.isfinite(obs.vector()).all())):
            self.reset()
            return None
        if self.last_time is not None and obs.timestamp <= self.last_time:
            return self.menu.selected
        dt = 0 if self.last_time is None else obs.timestamp-self.last_time
        if dt > self.s.max_gap:
            self.reset()
            dt = 0
        self.last_time = obs.timestamp
        self.tracking = True
        self.face_points=obs.face_points

        self.eyes_open = min(obs.left_ear, obs.right_ear) > max(0.065, self.calibration.ear_open*0.45)
        if paused:
            self._cancel_fixation()
            self.menu.cancel(clear=True)
            self.nav_side=None;self.nav_elapsed=self.nav_progress=0.
            self.previous_nav_supported=False
            self.pending_target=self.pending_since=self.stable_target=None
            return None
        if not self.eyes_open or not obs.gaze_valid:

            self.raw_window.clear();self.vertical_window.clear();self.feature_window.clear();self.sample_times.clear()
            self.previous_valid_target = None
            self.pending_target = self.pending_since = None
            self.release_elapsed = 0
            self.previous_nav_supported=False
            if self.last_valid_time is None or obs.timestamp-self.last_valid_time > self.s.fixation_grace:
                self._cancel_fixation()
                self.menu.cancel(clear=True)
                self.nav_side=None;self.nav_elapsed=self.nav_progress=0.
            return self.menu.selected
        self.last_valid_time = obs.timestamp
        self.raw_window.append(obs.gaze_x)
        self.vertical_window.append(obs.gaze_y)
        self.feature_window.append(obs.vector())
        self.sample_times.append(obs.timestamp)
        mapping=self.calibration.mapping or {}
        direct=mapping.get('classifier')
        paged=mapping.get('horizontal') is not None
        if paged:


            while len(self.sample_times)>3 and obs.timestamp-self.sample_times[0]>self.s.tracking_window_seconds+1e-9:
                self.sample_times.popleft();self.raw_window.popleft();self.vertical_window.popleft();self.feature_window.popleft()
        self.quality_message=''
        x, y = self.calibration.map_pair(median(self.raw_window), median(self.vertical_window),
                                         np.median(self.feature_window, axis=0))
        self.x = self.filter.update(x, obs.timestamp)
        self.y = self.vertical_filter.update(y, obs.timestamp)
        measured = self.menu.hit(self.x, self.y)
        if paged:
            self.y=.5
            side=navigation_at(self.x)


            instant_x=self.calibration.map_pair(obs.gaze_x,obs.gaze_y,obs.vector())[0]
            ready=len(self.raw_window)>=min(3,self.s.tracking_median)
            supported_side=ready and navigation_at(x)==side and navigation_at(instant_x)==side
            enabled=(side=='next' and self.page==0) or (side=='previous' and self.page==1)
            if side:
                self._cancel_fixation();self.menu.cancel(clear=True)
                if side!=self.nav_side:self.nav_side=side;self.nav_elapsed=0.
                elif enabled and self.nav_blocked!=side and supported_side and self.previous_nav_supported:self.nav_elapsed+=dt
                if not supported_side:self.nav_elapsed=0.
                self.previous_nav_supported=supported_side
                self.nav_progress=min(1.,self.nav_elapsed/self.s.navigation_seconds)
                if enabled and self.nav_progress>=1:
                    self.page=1-self.page;self.nav_blocked=side;self.nav_elapsed=self.nav_progress=0.
                    self.stable_target=self.pending_target=None;self.blocked=None
                return None
            self.nav_side=None;self.nav_elapsed=self.nav_progress=0.;self.nav_blocked=None
            self.previous_nav_supported=False
            local=card_at(self.x)
            supported=ready and local is not None and card_at(x)==local and card_at(instant_x)==local
            measured=local+self.page*4 if supported else None
        if direct is not None:
            from card_classifier import classify
            measured=classify(np.median(self.feature_window,axis=0),direct)
            if measured is None:self.quality_message='Fixe o olhar no cartão desejado.'
        if measured != self.pending_target:
            self.pending_target, self.pending_since = measured, obs.timestamp
        if measured is not None and measured != self.stable_target:
            row_change = self.stable_target is not None and measured//4 != self.stable_target//4
            delay = 0 if self.stable_target is None else (self.s.row_switch if row_change else self.s.column_switch)
            if self.pending_since is not None and obs.timestamp-self.pending_since >= delay:
                self.stable_target = measured
        target = measured if measured == self.stable_target else None
        if self.blocked is not None:
            if measured != self.blocked:
                self.release_elapsed += dt
                if self.release_elapsed >= self.s.repeat_release:
                    self.blocked = None
                    self.release_elapsed = 0
            else:
                self.release_elapsed = 0
        if target is None:

            self.previous_valid_target = None
            if self.last_supported is None or obs.timestamp-self.last_supported > self.s.fixation_grace:
                self._cancel_fixation()
                self.menu.cancel(clear=True)
            return self.menu.selected
        self.last_supported = obs.timestamp
        if paged:selected=self.menu.update_target(target,obs.timestamp)
        else:
            menu_x,menu_y=block_center(target) if direct is not None else (self.x,self.y)
            selected=self.menu.update(menu_x,obs.timestamp,menu_y)
        if target != self.fixation:
            self.fixation, self.fixation_elapsed = target, 0.0
        elif target is not None and target == self.previous_valid_target and target != self.blocked:
            self.fixation_elapsed += dt
        self.previous_valid_target = target
        if target is None or target == self.blocked:
            self.fixation_elapsed = 0
        self.confirm_progress = min(1.0, self.fixation_elapsed/self.s.confirmation_seconds)
        if selected is not None and target != self.blocked and self.confirm_progress >= 1-1e-9:
            self.last_confirmation = self.history.add(target, obs.timestamp)
            self.blocked = target
            self.fixation_elapsed = 0
            self.confirm_progress = 1.0
        return selected
