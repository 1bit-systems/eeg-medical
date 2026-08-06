"""Cross-subject raw EEGNet + band-pass (8-30 Hz), tuned: AdamW + LR schedule + more epochs.
The honest generalization test; band-pass is standard MI processing, not leakage.
"""
import warnings, time; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from mne.filter import filter_data
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0); np.random.seed(0)
t0=time.time()
ds=BNCI2014_001(); paradigm=MotorImagery(events=["left_hand","right_hand"],n_classes=2)
all_subj=[1,2,3,4,5,6,7,8,9]; SR=250; NSAMP=1001
BAND=(8,30)

def fetch(subjects):
    X,y,meta=paradigm.get_data(ds,subjects=subjects)
    return filter_data(X,sfreq=SR,l_freq=BAND[0],h_freq=BAND[1],verbose=0),\
           (y=="right_hand").astype(np.int64),np.asarray(meta["subject"])

class EEGNet1D(nn.Module):
    def __init__(self,n_ch,n_samp):
        super().__init__(); F1,D,F2=8,2,16
        self.net=nn.Sequential(
            nn.Conv2d(1,F1,(1,64),padding=(0,32),bias=False),nn.BatchNorm2d(F1),
            nn.Conv2d(F1,F1*D,(n_ch,1),groups=F1,bias=False),nn.BatchNorm2d(F1*D),nn.ELU(),
            nn.AvgPool2d((1,4)),nn.Dropout(0.25),
            nn.Conv2d(F1*D,F1*D,(1,16),padding=(0,8),groups=F1*D,bias=False),
            nn.Conv2d(F1*D,F2,1,bias=False),nn.BatchNorm2d(F2),nn.ELU(),
            nn.AvgPool2d((1,8)),nn.Dropout(0.25),
            nn.Flatten(),nn.Linear(F2*max(1,n_samp//32),2))
    def forward(self,x): return self.net(x.unsqueeze(1))

def train_eval(train_subj,test_subj,epochs=40):
    Xtr,ytr,_=fetch(train_subj); Xte,yte,_=fetch([test_subj])
    n_ch,n_samp=Xtr.shape[1],Xtr.shape[2]
    sc=StandardScaler().fit(Xtr.reshape(len(Xtr),-1))
    Xtr=sc.transform(Xtr.reshape(len(Xtr),-1)).reshape(-1,n_ch,n_samp)
    Xte=sc.transform(Xte.reshape(len(Xte),-1)).reshape(-1,n_ch,n_samp)
    m=EEGNet1D(n_ch,n_samp)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    lf=nn.CrossEntropyLoss()
    Xtr_t=torch.tensor(Xtr,dtype=torch.float32); ytr_t=torch.tensor(ytr)
    Xte_t=torch.tensor(Xte,dtype=torch.float32); yte_t=torch.tensor(yte)
    bs=48
    for e in range(epochs):
        m.train(); perm=torch.randperm(len(Xtr_t))
        for i in range(0,len(perm),bs):
            idx=perm[i:i+bs]; opt.zero_grad()
            lf(m(Xtr_t[idx]),ytr_t[idx]).backward(); opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        return (m(Xte_t).argmax(1)==yte_t).float().mean().item()

accs={}
print(f"[{time.time()-t0:4.0f}s] cross-subject raw EEGNet + 8-30Hz band-pass (AdamW, cosine), 40 ep...")
for held in all_subj:
    tr=[s for s in all_subj if s!=held]
    accs[held]=train_eval(tr,held)
    print(f"[{time.time()-t0:4.0f}s]  held-out subject {held}: {accs[held]*100:5.1f}%  "
          f"(running mean {np.mean(list(accs.values()))*100:.1f}%)")

print(f"\ncross-subject EEGNet + band-pass: mean={np.mean(list(accs.values()))*100:.1f}%  "
      f"(was 66.4% without band-pass)")
