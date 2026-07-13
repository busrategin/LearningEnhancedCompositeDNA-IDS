# ==============================================================================
# Copyright (c) 2026 CentraleSupelec (France) & Bilkent University (Turkey).
# All rights reserved.
#
# This source code is part of the official implementation for the paper:
# "Learning-Enhanced Composite DNA Data Storage Under Sampling Randomness and IDS Errors" by Busra Tegin and Tolga M Duman
#
# Funding Acknowledgment:
# This research was supported by the European Research Council (ERC). 
# Part of this work was conducted under the framework of the ERC Project 
# hosted at Bilkent University.
#
# Licensed under the MIT License. See the LICENSE file in the project root 
# for full license information.
# ==============================================================================# ==============================================================================
# NOTE ON SUBSTITUTION SIMULATION VS. PAPER THEORY:
# In Section III-D of the paper, the channel error model is defined as an 
# asymmetric substitution (where a base can only transition to the other 3 bases).
#
# However, to facilitate elegant tensor operations,
# this simulation utilizes the equivalent 4-ary symmetric uniform 
# channel via 'rng.integers(1, 5)' (where a base can transition to all 4 bases, 
# including itself, with equal probability).
# Hence the ps values taken in this simulation corresponds to 3*ps/4 in the paper
# This is why throughout Figs. 5 to 9, we have p_d = p_i = \frac{4}{3}p_s for the conversion

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import os



class RhoNetBiGRU(nn.Module):
    def __init__(self, hidden=256, layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=4,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Linear(2*hidden, 8*4)

    def forward(self, X):  # [B,160,4]
        denom = X.sum(dim=-1, keepdim=True).clamp_min(1.0)
        x = X / denom
        h, _ = self.gru(x)               # [B,160,2H]
        logits = self.fc(h).view(X.size(0), 160, 8, 4)
        return F.softmax(logits, dim=-1)



ss_np = np.array([
    [1,   0,   0,   0],
    [0,   1,   0,   0],
    [0,   0,   1,   0],
    [0,   0,   0,   1],
    [0.5, 0.5, 0,   0],
    [0,   0,   0.5, 0.5],
    [0.5, 0,   0.5, 0],
    [0,   0.5, 0,   0.5],
], dtype=np.float64)


def bits_to_states(bits_480: np.ndarray) -> np.ndarray:
    b = bits_480.reshape(160, 3).astype(np.int64)
    return (4*b[:, 0] + 2*b[:, 1] + b[:, 2]).astype(np.int64)


def sample_counts_from_ss(states: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((160, 4), dtype=np.int32)
    for i in range(160):
        out[i, :] = rng.multinomial(K, ss_np[states[i], :])
    return out


def sample_without_replacement_from_counts(d_obs: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    n = d_obs.shape[0]
    d_sampled = np.zeros((n, K), dtype=np.int16)
    for t in range(n):
        pool = np.repeat(np.arange(1, 5, dtype=np.int16), d_obs[t, :])
        rng.shuffle(pool)
        d_sampled[t, :] = pool[:K]
    return d_sampled

def apply_ids_channel_per_trace(d_sampled: np.ndarray, pd: float, pi: float, ps_prime: float,
                                rng: np.random.Generator) -> np.ndarray:
    n, K = d_sampled.shape
    d_IDall = np.zeros((2 * n, K), dtype=np.int16)

    for kk in range(K):
        err_rnd = rng.random(n)
        err_ind = np.zeros(n, dtype=np.int8)
        err_ind[err_rnd < pd] = 1
        err_ind[(err_rnd >= pd) & (err_rnd < pd + pi)] = -1
        err_ind[(err_rnd >= pd + pi) & (err_rnd < pd + pi + ps_prime)] = -2

        out = []
        col = d_sampled[:, kk]
        for kx in range(n):
            if err_ind[kx] == 1:          # deletion
                continue
            elif err_ind[kx] == -1:       # insertion then correct
                out.append(rng.integers(1, 5))
                out.append(int(col[kx]))
            elif err_ind[kx] == -2:       # substitution
                out.append(rng.integers(1, 5))
            else:                          # correct
                out.append(int(col[kx]))

        out = np.asarray(out, dtype=np.int16)
        d_IDall[:out.size, kk] = out

    return d_IDall

def normalize_short_long(d_IDall: np.ndarray, n: int):
    _, K = d_IDall.shape
    d_ID = np.zeros((n, K), dtype=np.int16)
    d_ID_short = np.zeros((n, K), dtype=np.int16)
    d_ID_long = np.zeros((2 * n, K), dtype=np.int16)

    ind_short, ind_long = [], []
    for k in range(K):
        if d_IDall[n - 1, k] == 0:
            d_ID_short[:, k] = d_IDall[:n, k]
            ind_short.append(k)
        elif d_IDall[n, k] > 0:
            d_ID_long[:, k] = d_IDall[:, k]
            ind_long.append(k)
        else:
            d_ID[:, k] = d_IDall[:n, k]


    d_ID_long2 = d_ID_long[:n, :].copy()
    for k in ind_long:
        ld = d_ID_long[:, k]
        nz = np.nonzero(ld)[0]
        if nz.size == 0:
            continue
        last = nz[-1]
        ld = ld[:last + 1]
        idx = np.round(np.linspace(0, last, n)).astype(int)
        d_ID_long2[:, k] = ld[idx]

    d_ID_short2 = d_ID_short.copy()
    for k in ind_short:
        ld = d_ID_short[:, k]
        nz = np.nonzero(ld)[0]
        if nz.size == 0:
            continue
        last = nz[-1]
        ld = ld[:last + 1]
        idx = np.round(np.linspace(0, last, n)).astype(int)
        d_ID_short2[:, k] = ld[idx]

    return d_ID, d_ID_short2, d_ID_long2

def split_short_long_raw(d_IDall: np.ndarray, n: int):

    _, K = d_IDall.shape
    d_ID = np.zeros((n, K), dtype=np.int16)
    d_ID_short = np.zeros((n, K), dtype=np.int16)
    d_ID_long = np.zeros((n, K), dtype=np.int16)

    for k in range(K):
        if d_IDall[n - 1, k] == 0:
            d_ID_short[:, k] = d_IDall[:n, k]
        elif d_IDall[n, k] > 0:
            d_ID_long[:, k] = d_IDall[:n, k]
        else:
            d_ID[:, k] = d_IDall[:n, k]
    return d_ID, d_ID_short, d_ID_long

def counts_per_position(d_ID: np.ndarray) -> np.ndarray:
    n, _ = d_ID.shape
    out = np.zeros((n, 4), dtype=np.int16)
    for t in range(n):
        bc = np.bincount(d_ID[t, :], minlength=5)
        out[t, :] = bc[1:5]
    return out


def generate_one_example(
    K=14,
    pd=0.003, pi=None, ps=None,
    resample_short_long=True,
    rng=None,
    csi_robust=False,
    pmin=0.001,
    pmax=0.005
):
    rng = np.random.default_rng() if rng is None else rng

    
    if csi_robust:
        pd = rng.uniform(pmin, pmax)
        pi = rng.uniform(pmin, pmax)
        ps = rng.uniform(pmin, pmax)
    else:
        if pi is None: pi = pd
        if ps is None: ps = pd

  
    ps_prime = (1.0 - pd - pi) * ps
   
    if ps_prime < 0:
        ps_prime = 0.0

    bits = rng.integers(0, 2, size=(480,), dtype=np.int64)
    states = bits_to_states(bits)

    d_obs = sample_counts_from_ss(states, K=K, rng=rng)
    d_sampled = sample_without_replacement_from_counts(d_obs, K, rng=rng)

    d_IDall = apply_ids_channel_per_trace(d_sampled, pd, pi, ps_prime, rng=rng)

    n = 160
    if resample_short_long:
        d_ID, d_sh, d_lo = normalize_short_long(d_IDall, n)
    else:
        d_ID, d_sh, d_lo = split_short_long_raw(d_IDall, n)

    X = (counts_per_position(d_ID) + counts_per_position(d_sh) + counts_per_position(d_lo)).astype(np.float32)
    y = bits.astype(np.float32)

    # Optional: return channel params too (useful for debugging)
    return X, y, (pd, pi, ps)


# dataset
def build_dataset(
    N=100_000,
    K=14, # number of reads (n in the paper)
    pd=0.003,
    resample_short_long=True,
    seed=0,
    csi_robust=False,
    pmin=0.001,
    pmax=0.005,
    return_params=False
):
    rng = np.random.default_rng(seed)
    X_all = np.zeros((N, 160, 4), dtype=np.float32)
    y_all = np.zeros((N, 480), dtype=np.float32)

    params = None
    if return_params:
        params = np.zeros((N, 3), dtype=np.float32)  # pd,pi,ps

    for i in range(N):
        X, y, (pdi, pii, psi) = generate_one_example(
            K=K, pd=pd,
            resample_short_long=resample_short_long,
            rng=rng,
            csi_robust=csi_robust,
            pmin=pmin, pmax=pmax
        )
        X_all[i] = X
        y_all[i] = y
        if return_params:
            params[i] = (pdi, pii, psi)

    if return_params:
        return X_all, y_all, params
    return X_all, y_all


class CompositeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # float32
        self.y = torch.from_numpy(y)  # float32
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# model
class RhoNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 8 * 4)

    def forward(self, X):  # [B,160,4]
        denom = X.sum(dim=-1, keepdim=True).clamp_min(1.0)
        x = X / denom
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x).view(X.size(0), 160, 8, 4)
        return F.softmax(logits, dim=-1)

def build_u_bits(device):
    u = torch.zeros(8, 3, device=device)
    for s in range(8):
        u[s, 0] = (s >> 2) & 1
        u[s, 1] = (s >> 1) & 1
        u[s, 2] = (s >> 0) & 1
    return u

def llr_from_counts_and_rho(X_counts, rho_hat, eps=1e-12):
    u_bits = build_u_bits(X_counts.device)  # [8,3]
    log_rho = torch.log(rho_hat.clamp_min(eps))          # [B,160,8,4]
    ll = (X_counts.unsqueeze(2) * log_rho).sum(dim=-1)   # [B,160,8]

    llrs = []
    for ell in range(3):
        mask0 = (u_bits[:, ell] == 0)
        mask1 = ~mask0
        a0 = torch.logsumexp(ll[:, :, mask0], dim=-1)
        a1 = torch.logsumexp(ll[:, :, mask1], dim=-1)
        llrs.append(a0 - a1)
    return torch.stack(llrs, dim=-1)  # [B,160,3]



def train_on_dataset(
    X_all,
    y_all,
    batch_size=64,
    epochs=250,
    lr=1e-3,
    device=None,
    save_path="model.pt",   # full path allowed
    resample_short_long=True,
    K=14,
    pd=0.003,

    val_fraction=0.1,
    split_seed=12345,
    num_workers=0,
    drop_last_train=True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"



    N = X_all.shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(idx)

    n_val = int(round(val_fraction * N))
    n_val = max(1, n_val)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_va, y_va = X_all[val_idx], y_all[val_idx]

    ds_tr = CompositeDataset(X_tr, y_tr)
    ds_va = CompositeDataset(X_va, y_va)

    pin = device.startswith("cuda")

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        num_workers=num_workers,
        pin_memory=pin
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin
    )


    model = RhoNetBiGRU(hidden=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    steps_per_epoch = len(dl_tr)

    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)


    base = os.path.basename(save_path)
    if base.endswith(".pt"):
        base = base[:-3]
    prefix = os.path.join(save_dir, base)

    print(f"\nDevice: {device}")
    if device.startswith("cuda"):
        print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 70)
    print(f"Dataset: N={N} | train={len(ds_tr)} | val={len(ds_va)} | val_fraction={val_fraction}")
    print(f"Split seed: {split_seed}")
    print("=" * 70)


    best_val_loss = float("inf")
    best_epoch = -1
    best_path = f"{prefix}_best.pt"

    for ep in range(1, epochs + 1):
        print(f"Epoch {ep}/{epochs}")
        model.train()

        epoch_start = time.time()
        step_times = []

        pbar = tqdm(
            enumerate(dl_tr, start=1),
            total=steps_per_epoch,
            bar_format="{n_fmt}/{total_fmt} [{bar:20}] {elapsed} {postfix}"
        )

   
        for step, (Xb, yb) in pbar:
            step_start = time.time()

            if pin:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
            else:
                Xb = Xb.to(device)
                yb = yb.to(device)

            rho_hat = model(Xb)                      
            llr = llr_from_counts_and_rho(Xb, rho_hat)  

     
            logits = -llr.reshape(Xb.size(0), 480)    

            loss = crit(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = (logits > 0).float()
                ber = (pred != yb).float().mean().item()

            step_times.append(time.time() - step_start)
            avg_ms = 1000 * np.mean(step_times[-50:])
            pbar.set_postfix_str(
                f"{avg_ms:.0f}ms/step - loss: {loss.item():.4e} - ber: {ber:.4e}"
            )

        train_time = time.time() - epoch_start
        print(f"Train epoch {ep} finished in {train_time/60:.2f} min")


        model.eval()
        val_loss_sum = 0.0
        val_bits = 0
        val_err_bits = 0

        with torch.no_grad():
            for Xv, yv in dl_va:
                if pin:
                    Xv = Xv.to(device, non_blocking=True)
                    yv = yv.to(device, non_blocking=True)
                else:
                    Xv = Xv.to(device)
                    yv = yv.to(device)

                rho_hat_v = model(Xv)
                llr_v = llr_from_counts_and_rho(Xv, rho_hat_v)
                logits_v = -llr_v.reshape(Xv.size(0), 480)

               
                v_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits_v, yv, reduction="sum"
                )
                val_loss_sum += float(v_loss.item())
                val_bits += int(yv.numel())

                v_pred = (logits_v > 0).float()
                val_err_bits += int((v_pred != yv).sum().item())

        val_loss = val_loss_sum / max(1, val_bits)
        val_ber = val_err_bits / max(1, val_bits)
        print(f"Validation: val_loss={val_loss:.6g}  val_ber={val_ber:.6g}")


        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": ep,
            "epochs": epochs,
            "batch_size": batch_size,
            "K": K, # number of reads n 
            "pd": pd,
            "resample_short_long": resample_short_long,
            "lr": lr,
            # --- NEW metadata ---
            "val_fraction": val_fraction,
            "split_seed": split_seed,
            "train_size": len(ds_tr),
            "val_size": len(ds_va),
            "val_loss": float(val_loss),
            "val_ber": float(val_ber),
            "best_val_loss_so_far": float(best_val_loss),
            "best_epoch_so_far": int(best_epoch),
        }


        epoch_path = f"{prefix}_ep{ep:03d}.pt"
        torch.save(ckpt, epoch_path)


        latest_path = f"{prefix}_latest.pt"
        torch.save(ckpt, latest_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep
            ckpt_best = dict(ckpt)
            ckpt_best["best_val_loss_so_far"] = float(best_val_loss)
            ckpt_best["best_epoch_so_far"] = int(best_epoch)
            best_path = f"{prefix}_best_ep{ep:03d}_val_loss{val_loss:.6g}.pt"

            torch.save(ckpt_best, best_path)
            print(f"Saved BEST: {best_path}  (epoch={ep}, val_loss={val_loss:.6g})")

        print(f"Saved: {epoch_path}  (and {latest_path})")

    print("=" * 70)
    print("Training completed.")
    print(f"BEST:   {best_path}  (epoch={best_epoch}, val_loss={best_val_loss:.6g})")
    print(f"LATEST: {prefix}_latest.pt")
    print("=" * 70)

    return model


if __name__ == "__main__":

    N = 100000. # dataset size
    K = 14 # number of reads which is n in the paper
    batch_size = 64
    epochs = 100


    Xr, yr = build_dataset(
        N=N, K=K,
        resample_short_long=True,
        seed=0,
        csi_robust=True,
        pmin=0.001,
        pmax=0.01
    )

    model_resampled = train_on_dataset(
        Xr, yr,
        batch_size=batch_size,
        epochs=epochs,
        resample_short_long=True,
        K=K,
        pd=-1,
        save_path="rhoNet-CSIRobust001-01_K14.pt",
        # NEW:
        val_fraction=0.1,
        split_seed=12345
    )
