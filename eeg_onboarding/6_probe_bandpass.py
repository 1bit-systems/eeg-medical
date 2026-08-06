import warnings, time; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from mne.filter import filter_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

torch.manual_seed(0); np.random.seed(0)
ds = BNCI2014_001()
paradigm = MotorImagery(events=["left_hand","right_hand"], n_classes=2)
X, y, _ = paradigm.get_data(ds, subjects=[1])
y = (y=="right_hand").astype(np.int64)
n,sr = X.shape[1], 250

def prep(X, band):
    if band:
        X = filter_data(X, sfreq=sr, l_freq=band[0], h_freq=band[1], verbose=0)
    n_t = X.shape[0]
    sc = StandardScaler().fit(X.reshape(n_t,-1))
    return sc.transform(X.reshape(n_t,-1)).reshape(-1,n,sr*4+1)

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

def run(band, epochs=40):
    Xp = prep(X, band)
    Xtr,Xte,ytr,yte = train_test_split(Xp,y,test_size=0.2,stratify=y,random_state=0)
    m=EEGNet1D(n,sr*4+1); opt=torch.optim.Adam(m.parameters(),lr=5e-3)
    lf=nn.CrossEntropyLoss()
    Xtr_t=torch.tensor(Xtr,dtype=torch.float32); Xte_t=torch.tensor(Xte,dtype=torch.float32)
    yte_t=torch.tensor(yte,dtype=torch.long)
    for e in range(epochs):
        m.train(); perm=torch.randperm(len(Xtr_t))
        for i in range(0,len(perm),32):
            idx=perm[i:i+32]
            opt.zero_grad(); loss=lf(m(Xtr_t[idx]),torch.tensor(ytr[idx],dtype=torch.long)); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): acc=(m(Xte_t).argmax(1)==yte_t).float().mean().item()
    return acc

for band,label in [(None,"full-band"),((8,30),"mu/beta 8-30Hz"),((4,40),"4-40 Hz")]:
    a=run(band); print(f"subject 1, EEGNet, {label:16s}: test={a*100:.1f}%")
