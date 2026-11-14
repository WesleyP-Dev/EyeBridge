"""Calibração e interpolação apenas do eixo horizontal, validado por repetição."""
import numpy as np


def card_bounds(column):
    """Limites normalizados compartilhados pelo desenho e pela seleção."""
    return .10+column*.2+.009, .10+(column+1)*.2-.009


def card_at(x):
    for column in range(4):
        left, right = card_bounds(column)
        if left <= x <= right:
            return column
    return None


def navigation_at(x):
    if -.15 <= x <= .08:
        return 'previous'
    if .92 <= x <= 1.15:
        return 'next'
    return None


def target_at(x):
    side = navigation_at(x)
    if side:
        return 0 if side == 'previous' else 5
    column = card_at(x)
    return None if column is None else column+1


def position(raw,model):
    a=np.asarray(model['anchors']);p=np.asarray(model['positions'])
    if a[-1]<a[0]:a=a[::-1];p=p[::-1]
    k=int(np.clip(np.searchsorted(a,raw)-1,0,len(a)-2))
    return float(p[k]+(raw-a[k])/(a[k+1]-a[k])*(p[k+1]-p[k]))


class HorizontalCalibrator:
    def __init__(self,settings):
        self.s=settings
        self.ids=(0,1,2,3,4,5,3,0,5,2,4,1)
        labels=('BORDA ESQUERDA','ÁGUA','COMIDA','BANHEIRO','CONFORTO','BORDA DIREITA')
        self.xs=(.04,.2,.4,.6,.8,.96)
        self.targets=tuple(labels[i] for i in self.ids)
        self.positions=tuple((self.xs[i],.5) for i in self.ids)
        self.order=tuple(i-1 if 1<=i<=4 else None for i in self.ids)
        self.index=0;self.since=self.last_time=None
        self.samples=[];self.sample_ears=[];self.groups=[];self.ears=[];self.error=''
        self.diagnostic={'version':6,'app_version':'3.10','accepted':False,'targets':[],
                         'protocol':'horizontal_independent_revisit'}

    @property
    def target(self):return self.targets[self.index]
    @property
    def position(self):return self.positions[self.index]

    def update(self,obs,previous):
        if not np.isfinite(obs.timestamp):
            self.since=None;self.samples=[];self.sample_ears=[];return None
        if self.last_time is not None and obs.timestamp<=self.last_time:return None
        gap=self.last_time is not None and obs.timestamp-self.last_time>self.s.max_gap
        self.last_time=obs.timestamp
        if (gap or not obs.valid or not obs.gaze_valid
                or not np.isfinite((obs.gaze_x,obs.left_ear,obs.right_ear)).all()
                or min(obs.left_ear,obs.right_ear)<max(.065,previous.ear_open*.45)):
            self.since=None;self.samples=[];self.sample_ears=[];return None
        if self.since is None:self.since=obs.timestamp
        elapsed=obs.timestamp-self.since
        if elapsed>=self.s.calibration_settle:
            self.samples.append(obs.gaze_x)
            self.sample_ears.append(min(obs.left_ear,obs.right_ear))
        if elapsed<self.s.calibration_settle+self.s.calibration_collect or len(self.samples)<10:return None
        values=np.array(self.samples)
        if np.percentile(values,90)-np.percentile(values,10)>.035:
            self.error='Olhar instável: fixe o ponto amarelo.';self.samples=[];self.sample_ears=[];self.since=None;return None
        self.groups.append(values);self.ears.append(float(np.median(self.sample_ears)))
        self.diagnostic['targets'].append({'name':self.target,'readings':values.tolist(),'median':float(np.median(values))})
        self.samples=[];self.sample_ears=[];self.since=None;self.error='';self.index+=1
        if self.index<len(self.targets):return None
        return self.finish()

    def finish(self):
        """Repetição independente e cobertura dos limites efetivamente usados na tela."""
        first=[self.groups[self.ids[:6].index(i)] for i in range(6)]
        second=[self.groups[6+self.ids[6:].index(i)] for i in range(6)]
        anchors=np.array([np.median(g) for g in first]);steps=np.diff(anchors)
        if not ((steps>.008).all() or (steps<-.008).all()):
            raise ValueError('Alvos horizontais muito próximos. Refaça G com rosto iluminado.')
        widths=list(range(3,self.s.tracking_median+1,2))
        rates=[]
        for i,g in enumerate(second):
            window_rates=[]
            for n in widths:
                reads=np.array([np.median(g[j:j+n]) for j in range(len(g)-n+1)])
                predicted=np.argmin(abs(reads[:,None]-anchors),axis=1)
                window_rates.append(float(np.mean(predicted==i)) if len(reads) else 0.)
            rates.append(min(window_rates))
        self.diagnostic['validation_rates']=rates
        bad=[i for i,r in enumerate(rates) if r<.8]
        if bad:raise ValueError('Leitura horizontal não repetiu: '+', '.join(self.targets[self.ids.index(i)] for i in bad))
        final=[float(np.median(np.r_[a,b])) for a,b in zip(first,second)]
        from core import Calibration
        result = Calibration(calibrated=True,ear_open=float(np.median(self.ears)),mapping={
            'mean':[.5,.5],'scale':[1.,1.],'coef':[[0.,0.]]*3,
            'horizontal':{'anchors':final,'positions':list(self.xs)}}).validate()


        independent={'anchors':anchors,'positions':self.xs}
        self.diagnostic['independent_hit_rates'] = [min(self._hit_rate(g,i,independent,n) for n in widths) for i,g in enumerate(second)]
        model=result.mapping['horizontal']
        coverage=[min(self._hit_rate(g,i,model,n) for g in (a,b) for n in widths)
                  for i,(a,b) in enumerate(zip(first,second))]
        self.diagnostic['fit_hit_rates']=coverage
        self.diagnostic['fit_windows']=widths
        bad=[i for i,r in enumerate(coverage) if r<.8]
        if bad:
            raise ValueError('Leituras fora da área de '+', '.join(self.targets[self.ids.index(i)] for i in bad)+'. Refaça G.')
        self.diagnostic['accepted']=True
        return result

    @staticmethod
    def _hit_rate(group,target,model,window):
        hits=[target_at(position(float(np.median(group[j:j+window])),model))==target
              for j in range(len(group)-window+1)]
        return float(np.mean(hits)) if hits else 0.
