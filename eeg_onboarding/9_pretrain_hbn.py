"""Self-supervised pretraining on HBN RestingState corpus.
Autoencoder reconstructs channel-aware EEG windows. Encoder is a reusable feature
extractor for future fine-tuning (e.g. on the MI task). Extracted features saved
as .npy for reuse without retraining.
Efficient: band-pass + channel projection + MSE reconstruction on 2s windows.
"""
import os, warnings, glob, time
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn
import mne
from mne.filter import filter_data
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(4)  # avoid thread thrash on a 6.9k-param model
# ponytail: 4 threads; bump only if per-epoch cost is the bottleneck
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "4")
t0 = time.time()
HBN = "/home/bcloud/eeg/hbn_r1"
OUT = "/home/bcloud/eeg/pretrain"
os.makedirs(OUT, exist_ok=True)
N_SUB = 60          # load up to this many subjects
WIN_SEC = 2.0       # window length
SF = 500
BP = (4, 40)        # theta+alpha+beta, drops 50/60Hz line noise field
F1, D = 8, 16       # encoder channels

files = sorted(glob.glob(f"{HBN}/*/eeg/*RestingState_eeg.set"))[:N_SUB]
W = int(WIN_SEC * SF)
print(f"[{time.time()-t0:5.0f}s] {len(files)} subjects, window={WIN_SEC}s*{SF}Hz={W} samples", flush=True)

# ---- build/collect full dataset of scaled windows ----
Xs = []
for f in files:
    raw = mne.io.read_raw_eeglab(f, preload=True, verbose=0)
    X = raw.get_data().astype(np.float64)                 # (ch, samples)
    X = filter_data(X, sfreq=SF, l_freq=BP[0], h_freq=BP[1], verbose=0)
    X = (X - X.mean()) / (X.std() + 1e-9)
    nwin = X.shape[1] // W
    Xu = X[:, : nwin * W].reshape(nwin, X.shape[0], W)    # (win, ch, W)
    Xs.append(Xu)
X = np.concatenate(Xs, axis=0)                            # (total_windows, ch, W)
print(f"[{time.time()-t0:5.0f}s] ready windows: {X.shape}", flush=True)


# ---- autoencoder: channel-aware, lightweight (runs on CPU) ----
class Encoder(nn.Module):
    def __init__(self, ch, w):
        super().__init__()
        self.ch = ch
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),   # temporal
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1, (ch, 1), groups=F1, bias=False),        # spatial depthwise
            nn.BatchNorm2d(F1), nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Conv2d(F1, D, (1, 16), padding=(0, 8), groups=F1, bias=False),
            nn.Conv2d(D, D, 1, bias=False), nn.BatchNorm2d(D), nn.ELU(),
            nn.AvgPool2d((1, 8)),
        )
    def forward(self, x):                 # (B,ch,W) -> (B,D,1,W//32)
        return self.net(x.unsqueeze(1))


class Decoder(nn.Module):
    def __init__(self, ch, w, zp=W//4):
        super().__init__()
        # upsample: (B,D,1,W//32) -> reconstruct (B,1,ch,W)
        d1 = F1 * 2
        self.head = nn.Sequential(
            nn.ConvTranspose2d(D, D, (1, 8), padding=(0, 4)), nn.BatchNorm2d(D), nn.ELU(),
            nn.ConvTranspose2d(D, d1, (1, 4), stride=(1,4)), nn.BatchNorm2d(d1), nn.ELU(),
            nn.ConvTranspose2d(d1, F1, (1, 4), stride=(1,4)), nn.BatchNorm2d(F1), nn.ELU(),
            nn.Conv2d(F1, ch, (1, 1)),  # to channels
        )
        # final temporal resize to exact W
        self.ch = ch; self.w = w
    def forward(self, z):
        x = self.head(z)                       # (B,ch,1,W)
        x = nn.functional.interpolate(x, size=(1, self.w), mode="bilinear", align_corners=False)
        # (B,ch,1,W) -> (B,ch,W) by dropping height dim
        return x.squeeze(2)                    # (B,ch,W)


def run():
    n = X.shape[0]
    Xt = torch.tensor(X, dtype=torch.float32)
    idx = torch.randperm(n)
    split = int(n * 0.9)
    tr, va = Xt[idx[:split]], Xt[idx[split:]]
    ch, w = X.shape[1], X.shape[2]
    enc, dec = Encoder(ch, w), Decoder(ch, w)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    bs = 32
    npar = sum(p.numel() for p in list(enc.parameters()) + list(dec.parameters()))
    print(f"model params {npar:,} | train windows {len(tr)} | val {len(va)}", flush=True)
    for ep in range(20):
        enc.train(); dec.train()
        perm = torch.randperm(len(tr)); tot = 0.0
        for i in range(0, len(perm), bs):
            idxb = perm[i:i+bs]
            xb = tr[idxb]
            opt.zero_grad()
            rec = dec(enc(xb))
            loss = lossf(rec, xb)
            loss.backward(); opt.step()
            tot += loss.item() * len(idxb)
        enc.eval(); dec.eval()
        with torch.no_grad():
            vloss = lossf(dec(enc(va[:128])), va[:128]).item()
        if ep in (0, 9, 19):
            print(f"[{time.time()-t0:5.0f}s] ep {ep+1:2d} train_mse={tot/len(tr):.4f} val_mse={vloss:.4f}", flush=True)
    # save encoder weights + latent summary
    torch.save(enc.state_dict(), f"{OUT}/encoder.pth")
    with torch.no_grad():
        lat = enc(Xt).view(n, -1).numpy()
    np.save(f"{OUT}/latents.npy", lat)
    print(f"encoder saved -> {OUT}/encoder.pth ; latents {lat.shape} -> {OUT}/latents.npy")
    return vloss


if __name__ == "__main__":
    run()
