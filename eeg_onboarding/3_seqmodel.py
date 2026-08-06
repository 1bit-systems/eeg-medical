"""Zyphra angle: raw EEG trials -> sequence model (LSTM) in torch. No CSP/handcrafted features."""
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
X, y, meta = paradigm.get_data(ds, subjects=[1, 2, 3])  # 3 subjects, raw trials
y = (y == "right_hand").astype(np.int64)
print(f"[{time.time()-t0:5.0f}s] raw trials loaded: {X.shape} (trials x 22ch x {X.shape[2]} samples @250Hz)")

# Standardize each channel across all trials (standard preprocessing, not feature engineering)
n_tr, n_ch, n_samp = X.shape
Xf = X.reshape(n_tr, -1)
scaler = StandardScaler().fit(Xf)
Xf = scaler.transform(Xf).reshape(n_tr, n_ch, n_samp)

Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, stratify=y, random_state=0)


class RawLSTM(nn.Module):
    def __init__(self, n_ch, n_samp):
        super().__init__()
        self.n_ch = n_ch
        # cheap channel merge: mean across channels -> 1 signal, then LSTM over time
        self.rnn = nn.LSTM(1, hidden_size=16, batch_first=True, num_layers=1)
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        # x: (B, n_ch, n_samp) -> (B, n_samp) meaned channels -> (B, n_samp, 1)
        s = x.mean(dim=1).unsqueeze(-1)
        out, _ = self.rnn(s)
        return self.head(out[:, -1, :])


model = RawLSTM(n_ch, n_samp)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()

Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
yte_t = torch.tensor(yte, dtype=torch.long)
# train on full train set for a fixed number of epochs (small fast demo)
batch = 32
for epoch in range(30):
    model.train()
    perm = torch.randperm(len(Xtr_t))
    tot, corr = 0, 0
    for i in range(0, len(perm), batch):
        idx = perm[i:i+batch]
        xb = Xtr_t[idx]
        yb = torch.tensor(ytr[idx], dtype=torch.long)
        opt.zero_grad()
        out = model(xb)
        loss = lossf(out, yb)
        loss.backward()
        opt.step()
        tot += len(yb)
        corr += (out.argmax(1) == yb).sum().item()
    model.eval()
    with torch.no_grad():
        xte_t = torch.tensor(Xte, dtype=torch.float32)
        preds = model(xte_t).argmax(1)
        acc = (preds == yte_t).float().mean().item()
    if epoch in (0, 14, 29):
        print(f"[{time.time()-t0:5.0f}s] epoch {epoch+1:2d}  train_acc={corr/tot*100:5.1f}%  "
              f"test_acc={acc*100:5.1f}%")

print(f"\nraw-sequence LSTM, 30 epochs, test accuracy: {acc*100:.1f}%  (chance 50%)")
print("note: naive mean-of-channels baseline. Real raw models use channel dim properly (EEGNet/Conv), this just proves the sequence path works.")
