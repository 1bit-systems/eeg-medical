"""EEGNet-1D: channel-aware raw EEG model (Lawhern 2018) via plain pytorch.
Fixes #3's mistake (channel-mean LSTM) by treating channels as a real dimension.
Within-subject, 3 subjects, split train/test on the same subject as #3 for a fair A/B.
"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
np.random.seed(0)
t0 = time.time()

ds = BNCI2014_001()
paradigm = MotorImagery(events=["left_hand", "right_hand"], n_classes=2)
X, y, meta = paradigm.get_data(ds, subjects=[1, 2, 3])
y = (y == "right_hand").astype(np.int64)
n_tr, n_ch, n_samp = X.shape
print(f"[{time.time()-t0:5.0f}s] raw trials: {X.shape} ({n_tr} x {n_ch}ch x {n_samp} @250Hz)")

Xf = X.reshape(n_tr, -1)
scaler = StandardScaler().fit(Xf)
Xf = scaler.transform(Xf).reshape(n_tr, n_ch, n_samp)
Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, stratify=y, random_state=0)


class EEGNet1D(nn.Module):
    """Minimal Lawhern-2018 EEGNet for 1D (n_ch, n_samp) input."""
    def __init__(self, n_ch, n_samp):
        super().__init__()
        # F1 temporal filters, depthwise per-channel, separable conv
        F1, D, F2 = 8, 2, 16
        self.tconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),  # temporal
            nn.BatchNorm2d(F1),
        )
        self.dconv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),  # spatial (channel depthwise)
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
        )
        self.sconv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
        )
        in_flat = F2 * max(1, n_samp // 32)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_flat, 2),
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (B,1,ch,samp)
        x = self.tconv(x)
        x = self.dconv(x)
        x = self.sconv(x)
        return self.classifier(x)


model = EEGNet1D(n_ch, n_samp)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
lossf = nn.CrossEntropyLoss()
Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
Xte_t = torch.tensor(Xte, dtype=torch.float32)
yte_t = torch.tensor(yte, dtype=torch.long)
batch = 32

print(f"model params: {sum(p.numel() for p in model.parameters()):,}")
for epoch in range(30):
    model.train()
    perm = torch.randperm(len(Xtr_t))
    tot = corr = 0
    for i in range(0, len(perm), batch):
        idx = perm[i:i + batch]
        xb, yb = Xtr_t[idx], torch.tensor(ytr[idx], dtype=torch.long)
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()
        tot += len(yb)
        corr += (model(xb).argmax(1) == yb).sum().item()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte_t).argmax(1) == yte_t).float().mean().item()
    if epoch in (0, 14, 29):
        print(f"[{time.time()-t0:5.0f}s] epoch {epoch+1:2d}  train={corr/tot*100:5.1f}%  test={acc*100:5.1f}%")

print(f"\nEEGNet-1D raw, within-subject test accuracy: {acc*100:.1f}%  (vs LSTM chance ~50%)")
