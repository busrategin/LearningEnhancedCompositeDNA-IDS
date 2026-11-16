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

import time
import numpy as np
from scipy.io import loadmat
import pandas as panda

import torch
import torch.nn as nn



class LLRNetBiGRU(nn.Module):
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
        self.fc = nn.Linear(2 * hidden, 3)  

    def forward(self, X): 
        denom = X.sum(dim=-1, keepdim=True).clamp_min(1.0)
        x = X / denom
        h, _ = self.gru(x)        
        z = self.fc(h)             
        return z


def load_llr_model(ckpt_path: str, device="cpu", hidden=256, layers=2, dropout=0.1):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = LLRNetBiGRU(hidden=hidden, layers=layers, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt


@torch.no_grad()
def model_to_llr_p0_over_p1(model: nn.Module, X_counts_t: torch.Tensor) -> torch.Tensor:
    """
    model outputs logits z = log(P1/P0).
    LDPC decoder expects LLR = log(P0/P1).
    So: LLR = -z
    X_counts_t: [B,160,4] float32
    returns:    [B,480] float32
    """# [B,160,3]
    z = model(X_counts_t)  # [B,160,3]
    z480 = z.reshape(X_counts_t.size(0), 480)
    return -z480



def ldpc_decode_logspa(H, llr, max_iters=50, llr_clip=50.0, early_stop=True):
    H = (np.asarray(H).astype(np.int8) & 1)
    m, n = H.shape

    llr = np.asarray(llr, dtype=np.float64)
    single = (llr.ndim == 1)
    if single:
        llr = llr[None, :]

    B, n2 = llr.shape
    if n2 != n:
        raise ValueError(f"LLR length {n2} does not match H columns {n}.")

    check_nbrs = [np.where(H[i] == 1)[0] for i in range(m)]
    q = [np.clip(llr[:, check_nbrs[i]], -llr_clip, llr_clip).copy() for i in range(m)]
    r = [np.zeros_like(q[i]) for i in range(m)]

    def hard_from_llr(L):
        return (L < 0).astype(np.int8)  

    def syndrome_bits(xb):
        return (xb @ H.T) & 1

    ATANH_CLIP = 1.0 - 1e-12
    iters_used = max_iters
    success = np.zeros(B, dtype=bool)
    L_post = llr.copy()

    for it in range(1, max_iters + 1):
        for i in range(m):
            qs = np.clip(q[i], -llr_clip, llr_clip)
            t = np.tanh(qs / 2.0)
            prod_all = np.prod(t, axis=1, keepdims=True)

            deg = t.shape[1]
            for k in range(deg):
                denom = t[:, [k]]                      
                mask = (np.abs(denom[:, 0]) < 1e-300) 

                prod_excl = np.empty_like(denom)       

                not_mask = ~mask
                prod_excl[not_mask, 0] = prod_all[not_mask, 0] / denom[not_mask, 0]

                if np.any(mask):
                    if deg == 1:
                        prod_excl[mask, 0] = 1.0
                    else:
                        t_ex = np.delete(t, k, axis=1)  # (B, deg-1)
                        prod_excl[mask, 0] = np.prod(t_ex[mask, :], axis=1)

                prod_excl = np.clip(prod_excl, -ATANH_CLIP, ATANH_CLIP)
                r[i][:, k] = 2.0 * np.arctanh(prod_excl)[:, 0]

            r[i] = np.clip(r[i], -llr_clip, llr_clip)

        L_post = llr.copy()
        for i in range(m):
            js = check_nbrs[i]
            L_post[:, js] += r[i]

        for i in range(m):
            js = check_nbrs[i]
            q[i] = np.clip(L_post[:, js] - r[i], -llr_clip, llr_clip)

        x_hat = hard_from_llr(L_post)
        syn = syndrome_bits(x_hat)
        ok = np.all(syn == 0, axis=1)

        success = ok
        if early_stop and np.all(ok):
            iters_used = it
            break

    x_hat = hard_from_llr(L_post)
    if single:
        return x_hat[0], L_post[0], iters_used, bool(success[0])
    return x_hat, L_post, iters_used, success



def bits_to_int_left_msb(bit_matrix: np.ndarray) -> np.ndarray:
    L = bit_matrix.shape[1]
    weights = (1 << np.arange(L - 1, -1, -1, dtype=np.int64))
    return (bit_matrix.astype(np.int64) * weights).sum(axis=1)

def ldpc_encode_systematic(x: np.ndarray, Gsys: np.ndarray) -> np.ndarray:
    return (x @ Gsys) & 1

def sample_composite_channel(d_sym: np.ndarray, ss: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    n = d_sym.size
    d_obs = np.zeros((n, 4), dtype=np.int32)
    for i in range(n):
        d_obs[i, :] = rng.multinomial(K, ss[d_sym[i], :])
    return d_obs

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
            if err_ind[kx] == 1:
                continue
            elif err_ind[kx] == -1:
                out.append(rng.integers(1, 5))
                out.append(int(col[kx]))
            elif err_ind[kx] == -2:
                out.append(rng.integers(1, 5))
            else:
                out.append(int(col[kx]))

        out = np.asarray(out, dtype=np.int16)
        d_IDall[:out.size, kk] = out

    return d_IDall

def normalize_short_long(d_IDall: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def counts_per_position(d_ID: np.ndarray) -> np.ndarray:
    n, _ = d_ID.shape
    out = np.zeros((n, 4), dtype=np.int16)
    for t in range(n):
        bc = np.bincount(d_ID[t, :], minlength=5)
        out[t, :] = bc[1:5]
    return out

def simulate_one_codeword_features(
    Gsys: np.ndarray,
    L: int,
    K: int,
    pd: float,
    pi: float,
    ps: float,
    rng: np.random.Generator,
    ss: np.ndarray,
    force_T: int = 160
) -> tuple[np.ndarray, np.ndarray]:
    k = Gsys.shape[0]
    x_info = rng.integers(0, 2, size=k, dtype=np.uint8)
    d = ldpc_encode_systematic(x_info, Gsys.astype(np.uint8))

    rem = d.size % L
    if rem != 0:
        d = np.concatenate([d, np.zeros(L - rem, dtype=d.dtype)])

    d_matrix = d.reshape(-1, L)
    d_sym = bits_to_int_left_msb(d_matrix)  
    T = d_sym.size

    d_obs = sample_composite_channel(d_sym.astype(np.int64), ss, K, rng)
    d_sampled = sample_without_replacement_from_counts(d_obs, K, rng)

    ps_prime = (1.0 - pd - pi) * ps
    d_IDall = apply_ids_channel_per_trace(d_sampled, pd, pi, ps_prime, rng)
    d_ID, d_ID_short, d_ID_long = normalize_short_long(d_IDall, T)

    og = counts_per_position(d_ID)
    sh = counts_per_position(d_ID_short)
    lo = counts_per_position(d_ID_long)
    X = (og + sh + lo).astype(np.float64)

    if force_T is not None and T != force_T:
        if T > force_T:
            X = X[:force_T, :]
        else:
            X_pad = np.zeros((force_T, 4), dtype=X.dtype)
            X_pad[:T, :] = X
            X = X_pad

    return X, x_info



def estimate_bler_streaming_pytorch_llrnet(
    ckpt_path: str,
    H_path: str,
    H_key: str,
    Gsys: np.ndarray,
    pd_list: list[float],
    L: int = 3,
    K: int = 14,
    max_ldpc_iters: int = 20,
    iter_total_map: dict[float, int] | None = None,
    batch_size: int = 2048,
    seed: int | None = None,
    save_results_xlsx: str | None = None,
    device: str = "cpu",
):
    rng = np.random.default_rng(seed)

    A = loadmat(H_path, squeeze_me=True, struct_as_record=False)
    H = A[H_key]
    n_codebits = H.shape[1]

    k_info = Gsys.shape[0]
    if Gsys.shape[1] != n_codebits:
        raise ValueError(f"Gsys has {Gsys.shape[1]} columns but H has {n_codebits} columns.")

    ss = np.array([
        [1,   0,   0,   0],
        [0,   1,   0,   0],
        [0,   0,   1,   0],
        [0,   0,   0,   1],
        [0.5, 0.5, 0,   0],
        [0,   0,   0.5, 0.5],
        [0.5, 0,   0.5, 0],
        [0,   0.5, 0,   0.5],
    ], dtype=np.float64)

    model, meta = load_llr_model(ckpt_path, device=device, hidden=256, layers=2, dropout=0.1)
    print("Loaded ckpt:", ckpt_path)
    if isinstance(meta, dict) and "epoch" in meta:
        print("  epoch:", meta["epoch"])

    results = {}
    pd_listx, bler_listx = [], []

    for pd in pd_list:
        pi = float(pd)
        ps = float(pd)

        if iter_total_map is not None and pd in iter_total_map:
            iter_total = int(iter_total_map[pd])
        else:
            if pd <= 0.0015:
                iter_total = 1000000
            elif pd < 0.00275:
                iter_total = 1000000
            elif pd < 0.0045:
                iter_total = 100000
            else:
                iter_total = 50000

        n_blocks = (iter_total + batch_size - 1) // batch_size
        err_blocks = 0
        done = 0

        t0 = time.time()
        for _ in range(n_blocks):
            B = min(batch_size, iter_total - done)

            Xb = np.zeros((B, 160, 4), dtype=np.float64)
            Ub = np.zeros((B, k_info), dtype=np.uint8)

            for i in range(B):
                X, u = simulate_one_codeword_features(
                    Gsys=Gsys,
                    L=L,
                    K=K,
                    pd=float(pd),
                    pi=float(pi),
                    ps=float(ps),
                    rng=rng,
                    ss=ss,
                    force_T=160
                )
                Xb[i, :, :] = X
                Ub[i, :] = u

            Xb_t = torch.from_numpy(Xb.astype(np.float32)).to(device)
            with torch.no_grad():
                llr = model_to_llr_p0_over_p1(model, Xb_t)  
            llr_np = llr.cpu().numpy().astype(np.float64)

            if llr_np.shape[1] != n_codebits:
                if llr_np.shape[1] > n_codebits:
                    llr_np = llr_np[:, :n_codebits]
                else:
                    pad = np.zeros((B, n_codebits), dtype=np.float64)
                    pad[:, :llr_np.shape[1]] = llr_np
                    llr_np = pad

            x_hat_code, _, _, _ = ldpc_decode_logspa(H, llr_np, max_iters=max_ldpc_iters)
            x_hat_info = x_hat_code[:, :k_info].astype(np.int16)
            u_true = Ub.astype(np.int16)

            err_blocks += int(np.sum(np.any(x_hat_info != u_true, axis=1)))
            done += B

        bler = err_blocks / iter_total
        results[float(pd)] = float(bler)
        dt = time.time() - t0
        print(f"pd={pd:.5f}  iter_total={iter_total}  BLER={bler:.6g}  time={dt/60:.2f} min")

        if save_results_xlsx is not None:
            pd_listx.append(float(pd))
            bler_listx.append(float(bler))
            df = panda.DataFrame({"pd": pd_listx, "BLER": bler_listx})
            df.to_excel(save_results_xlsx, index=False)
            print(f"Saved to {save_results_xlsx}")

    return results



if __name__ == "__main__":
    K = 14
    L = 3

    # pd sweep
    pds = np.arange(0.0015, 0.006 + 1e-12, 0.00025).tolist()

    # Paths
    ckpt_path = "LLRNet_CSI_robust001-005_K14_03-02-2026_best_ep018_val_loss0.023057.pt"
    H_path = "Hmatrices.mat"
    H_key = "H"
    Gsys = loadmat(H_path)["Gsys"].astype(np.uint8)

    results = estimate_bler_streaming_pytorch_llrnet(
        ckpt_path=ckpt_path,
        H_path=H_path,
        H_key=H_key,
        Gsys=Gsys,
        pd_list=pds,
        L=L,
        K=K, # number of reads: n in the paper
        max_ldpc_iters=20,
        batch_size=8096,
        seed=None,
        save_results_xlsx="/BLER_LLRNet_CSIRobust_IDS_pdSweep_K14.xlsx",
        device="cpu"
    )
