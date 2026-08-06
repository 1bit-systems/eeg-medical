"""How close to 100% can a STRONG subject get? Tests whether the ceiling is model-limited
or signal-limited. band-pass + tuned EEGNet, within-subject, cross-validated."""
import warnings,time; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from mne.filter import filter_data
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0); np.random.seed(0)
ds=BNCI2014_001(); paradigm=MotorImagery(events=["left_hand","right_hand"],n_classes=2)
SR=250; NSAMP=1001; BAND=(8,30)
SUBJ=[1,2,3,4,5,6,7,8,9]

def fetch(subjects):
    X,y,_=paradigm.get_data(ds,subjects=subjects)
    X=filter_data(X,sfreq=SR,l_freq=BAND[0],h_freq=BAND[1],verbose=0)
    return X,(y=="right_hand").astype(np.int64)

def model(n_ch,n_samp):
    F1,D,F2=8,2,16
    return nn.Sequential(
        nn.Unflatten(1, (1, n_ch)),  # (B,n_ch,n_samp)->(B,1,n_ch,n_samp)
        nn.Conv2d(1,F1,(1,64),padding=(0,32),bias=False),nn.BatchNorm2d(F1),
        nn.Conv2d(F1,F1*D,(n_ch,1),groups=F1,bias=False),nn.BatchNorm2d(F1*D),nn.ELU(),
        nn.AvgPool2d((1,4)),nn.Dropout(0.25),
        nn.Conv2d(F1*D,F1*D,(1,16),padding=(0,8),groups=F1*D,bias=False),
        nn.Conv2d(F1*D,F2,1,bias=False),nn.BatchNorm2d(F2),nn.ELU(),
        nn.AvgPool2d((1,8)),nn.Dropout(0.25),
        nn.Flatten(),nn.Linear(F2*max(1,n_samp//32),2))

def run(subject,epochs=60):
    X,y=fetch([subject]); n_ch,n_samp=X.shape[1],X.shape[2]
    X=StandardScaler().fit_transform(X.reshape(len(X),-1)).reshape(-1,n_ch,n_samp)
    # 5-fold CV within subject
    from sklearn.model_selection import StratifiedKFold
    kf=StratifiedKFold(5,shuffle=True,random_state=0)
    accs=[]
    Xt=torch.tensor(X,dtype=torch.float32); yt=torch.tensor(y)
    for tr,te in kf.split(X,y):
        m=model(n_ch,n_samp)
        opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-3)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
        lf=nn.CrossEntropyLoss()
        for e in range(epochs):
            m.train(); perm=torch.randperm(len(tr))
            for i in range(0,len(perm),48):
                idx=tr[perm[i:i+48]]
                opt.zero_grad(); lf(m(Xt[idx]),yt[idx]).backward(); opt.step()
            sched.step()
        m.eval()
        with torch.no_grad(): accs.append((m(Xt[te]).argmax(1)==yt[te]).float().mean().item())
    return np.mean(accs)

print(f"within-subject EEGNet + band-pass, 60 ep, 5-fold CV — probing the ceiling per subject:")
for s in SUBJ:
    a=run(s); print(f"  subject {s}: {a*100:.1f}%")
