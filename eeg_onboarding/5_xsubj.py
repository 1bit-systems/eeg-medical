"""Cross-subject raw EEGNet: train on some subjects, test on held-out subject.
This is the hard generalization test (the gap a foundation model must close).
Uses a single shared channel layout / resampling so cross-subject is meaningful.
"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0)
np.random.seed(0)
t0 = time.time()

ds = BNCI2014_001()
paradigm = MotorImagery(events=["left_hand", "right_hand"], n_classes=2)
all_subj = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def fetch(subjects):
    X, y, meta = paradigm.get_data(ds, subjects=subjects)
    y = (y == "right_hand").astype(np.int64)
    return X, y, np.asarray(meta["subject"])


class EEGNet1D(nn.Module):
    def __init__(self, n_ch, n_samp):
        super().__init__()
        F1, D, F2 = 8, 2, 16
        self.tconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.dconv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
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
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(F2 * max(1, n_samp // 32), 2))

    def forward(self, x):
        x = self.tconv(x.unsqueeze(1))
        x = self.sconv(self.dconv(x))
        return self.classifier(x)


def train_eval(train_subj, test_subj):
    Xtr, ytr, _ = fetch(train_subj)
    Xte, yte, _ = fetch([test_subj])
    n_ch, n_samp = Xtr.shape[1], Xtr.shape[2]
    sc = StandardScaler().fit(Xtr.reshape(len(Xtr), -1))
    Xtr = sc.transform(Xtr.reshape(len(Xtr), -1)).reshape(-1, n_ch, n_samp)
    Xte = sc.transform(Xte.reshape(len(Xte), -1)).reshape(-1, n_ch, n_samp)

    model = EEGNet1D(n_ch, n_samp)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    lossf = nn.CrossEntropyLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    Xte_t, yte_t = torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte)
    bs = 32
    for epoch in range(20):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte_t).argmax(1) == yte_t).float().mean().item()
    return acc


print(f"[{time.time()-t0:5.0f}s] leave-one-subject-out raw EEGNet (train 8, test 1)...")
accs = {}
for held in all_subj:
    train_subj = [s for s in all_subj if s != held]
    accs[held] = train_eval(train_subj, held)
    print(f"[{time.time()-t0:5.0f}s]  held-out subject {held}: {accs[held]*100:5.1f}%  "
          f"(running mean {np.mean(list(accs.values()))*100:.1f}%)")

print(f"\ncross-subject raw EEGNet: mean={np.mean(list(accs.values()))*100:.1f}%  "
      f"(chance 50%) — within-subject was ~79%; the gap is the generalization problem")
