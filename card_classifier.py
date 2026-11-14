"""Reconhece cartões diretamente; não tenta estimar pixels da tela."""
import numpy as np


def distances(vector, model):
    z=(np.asarray(vector,float)[:4]-model['mean'])/model['scale']
    diff=z-np.asarray(model['centers'])
    return np.einsum('ni,ij,nj->n',diff,np.asarray(model['precision']),diff)


def classify(vector, model):
    vector=np.asarray(vector,float)
    if vector.ndim!=1 or len(vector)<4 or not np.isfinite(vector[:4]).all():return None
    d=distances(vector,model)
    order=np.argsort(d)
    best=int(order[0])
    if d[best]>model['radius'][best] or d[order[1]]-d[best]<.25:
        return None
    return best-1 if best else None


def position(vector,model):
    """Interpolação contínua local das nove âncoras, sem encaixar no cartão vencedor."""
    d=np.maximum(0,distances(vector,model))
    coordinates=np.array([[.5,.5]]+[[((i%4)+.5)/4,((i//4)+.5)/2] for i in range(8)])
    closest=int(np.argmin(d))
    if d[closest]<1e-10:
        return tuple(float(v) for v in coordinates[closest])


    radius=max(float(np.sort(d)[4]),1e-6)+1e-6
    weights=np.maximum(0,1-d/radius)**2/(d+1e-6)
    point=weights@coordinates/weights.sum()
    return tuple(float(v) for v in point)


def fit(groups, window=5, validation_groups=None):
    groups=[np.asarray(g,float)[:,:4] for g in groups]
    if validation_groups is not None:
        validation_groups=[np.asarray(g,float)[:,:4] for g in validation_groups]
    training=groups if validation_groups is not None else [g[:len(g)//2] for g in groups]
    mean=np.median(np.vstack(training),axis=0)
    scale=np.maximum(np.std(np.vstack(training),axis=0),[.004,.03,.02,.01])

    def build(samples):
        centers=np.array([np.median(g,axis=0) for g in samples])
        residual=np.vstack([(g-center)/scale for g,center in zip(samples,centers)])
        covariance=np.cov(residual.T)
        covariance=.6*covariance+.4*np.diag(np.diag(covariance))+np.eye(4)*.03
        return dict(mean=mean.tolist(),scale=scale.tolist(),centers=((centers-mean)/scale).tolist(),
                    precision=np.linalg.inv(covariance).tolist(),radius=[100.]*9)

    model=build(training)
    rates=[];radii=[]
    for i,g in enumerate(groups):
        raw=np.asarray(validation_groups[i])[:,:4] if validation_groups is not None else g[len(g)//2:]
        samples=[np.median(raw[j:j+window],axis=0) for j in range(max(1,len(raw)-window+1))]
        ds=np.array([distances(v,model) for v in samples])
        winners=np.argmin(ds,axis=1)
        rates.append(float((winners==i).mean()))
        radii.append(float(max(9,np.percentile(ds[:,i],99)+6)))
    failed=[i for i,r in enumerate(rates) if r<.75]
    if failed:return None,dict(validation_rates=rates,failed_targets=failed)
    final=build([np.vstack((g,validation_groups[i]))[:,:4] for i,g in enumerate(groups)]
                if validation_groups is not None else groups)
    final['radius']=radii
    final['validation_protocol']='independent_revisit' if validation_groups is not None else 'same_target_split'
    return final,dict(validation_rates=rates,failed_targets=[])


def validate(model):
    shapes={'mean':(4,),'scale':(4,),'centers':(9,4),'precision':(4,4),'radius':(9,)}
    for key,shape in shapes.items():
        a=np.asarray(model[key],float)
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Classificador inválido')
    if (np.asarray(model['scale'])<=0).any() or (np.asarray(model['radius'])<=0).any():
        raise ValueError('Escala do classificador inválida')
    precision=np.asarray(model['precision'])
    if not np.allclose(precision,precision.T) or (np.linalg.eigvalsh(precision)<=0).any():
        raise ValueError('Métrica do classificador inválida')
