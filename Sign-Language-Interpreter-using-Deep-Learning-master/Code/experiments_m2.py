"""
Milestone 2 preliminary experiments (kept so the numbers in the Milestone 2 report can be reproduced).

The final versions of these ideas now live in features.py, augment.py, train_classifier.py and evaluate.py.

Reproduces Section 4.5 of the Milestone 1 & 2 report:
  - raw 63-D vs palm-size scaled vs engineered 88-D features (same SVM)
  - grid search for SVM C / gamma
  - MLP baseline
  - predict() vs argmax(predict_proba()) disagreement

Run from Code/:  .venv/bin/python experiments_m2.py
"""
import os
import time, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
_old_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'landmarks.csv')
if os.path.exists(_old_csv):
    # The report's numbers were produced from this file; its row order fixes the random split.
    df = pd.read_csv(_old_csv); y = df.label.values; X = df.drop(columns=['label']).to_numpy('float64')
else:
    # Same 63-D values from the committed raw CSV; rows are in a different order,
    # so the random split and the exact numbers differ slightly from the report.
    import data
    from features import features_v1
    _P, _meta = data.load_samples('letter', sources=('asl_now',))
    y = np.array([m['label'] for m in _meta]); X = features_v1(_P)
print(len(y), len(set(y)))
def scale(X):
    P=X.reshape(-1,21,3); s=np.linalg.norm(P[:,9,:2],axis=1,keepdims=True); s[s==0]=1
    return (P/s[:,:,None]).reshape(len(X),-1)
def eng(X):
    P=X.reshape(-1,21,3)/np.maximum(np.linalg.norm(X.reshape(-1,21,3)[:,9,:2],axis=1),1e-9)[:,None,None]
    tips=[4,8,12,16,20]; d=[np.linalg.norm(P[:,a]-P[:,b],axis=1) for i,a in enumerate(tips) for b in tips[i+1:]]
    chains=[(0,1,2),(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),(0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),(14,15,16),(0,17,18),(17,18,19),(18,19,20)]
    ang=[]
    for a,b,c in chains:
        u=P[:,a]-P[:,b]; v=P[:,c]-P[:,b]
        ang.append(np.arccos(np.clip((u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9),-1,1)))
    return np.hstack([P.reshape(len(X),-1),np.array(d).T,np.array(ang).T])
Xtr,Xte,ytr,yte=train_test_split(np.arange(len(X)),y,test_size=0.25,random_state=42,stratify=y)
cv=StratifiedKFold(5,shuffle=True,random_state=42)
for name,F in [('raw63',X),('scaled63',scale(X)),('engineered',eng(X))]:
    m=make_pipeline(StandardScaler(),SVC(C=10,gamma='scale',probability=True,random_state=42))
    m.fit(F[Xtr],ytr); p=m.predict(F[Xte])
    pa=m.classes_[m.predict_proba(F[Xte]).argmax(1)]
    c=cross_val_score(make_pipeline(StandardScaler(),SVC(C=10,gamma='scale')),F[Xtr],ytr,cv=cv)
    print(name,F.shape[1],'test acc %.3f macroF1 %.3f cv(train) %.3f±%.3f  predict!=argmax %d'%(accuracy_score(yte,p),f1_score(yte,p,average='macro'),c.mean(),c.std(),(p!=pa).sum()))
    if name=='raw63':
        t=time.perf_counter()
        for i in range(300): m.predict_proba(F[i:i+1])
        print(' latency ms',(time.perf_counter()-t)/300*1000)
g=GridSearchCV(make_pipeline(StandardScaler(),SVC()),{'svc__C':[1,3,10,30,100],'svc__gamma':['scale',0.003,0.01,0.03]},cv=cv,n_jobs=-1).fit(X[Xtr],ytr)
print('grid best',g.best_params_,'%.3f'%g.best_score_)
mlp=make_pipeline(StandardScaler(),MLPClassifier((128,64),alpha=1e-3,max_iter=600,early_stopping=True,random_state=42)).fit(X[Xtr],ytr)
print('mlp test %.3f'%accuracy_score(yte,mlp.predict(X[Xte])))
