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
# ==============================================================================
# 
# 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import numpy as np
from scipy.io import loadmat
from scipy.special import comb
import pandas as panda


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

u_bits_np = np.zeros((8, 3), dtype=np.int32)
for _s in range(8):
    u_bits_np[_s, 0] = (_s >> 2) & 1
    u_bits_np[_s, 1] = (_s >> 1) & 1
    u_bits_np[_s, 2] = (_s >> 0) & 1


def compute_rho_hat_E(ss, pd, pi, ps, E, K_hat):
    if K_hat == 0:
        return np.full((E, ss.shape[0], 4), 0.25)

    S = ss.shape[0]
    rho_hat = np.zeros((E, S, 4), dtype=np.float64)
    ps_hat = 4.0 * ps / 3.0   

    binom_E  = comb(E, 2, exact=False)
    denom_CE = (1 - pi - pd) ** 2 + 2 * pi * pd * binom_E

    t_arr       = np.arange(K_hat + 1)
    binom_coeff = np.array([comb(K_hat, t, exact=False) for t in t_arr])

    for i in range(1, E + 1):
        binom_Ei  = comb(E - i, 2, exact=False)
        binom_im1 = comb(i - 1, 2, exact=False)
        num_CE    = (1 - pi - pd) ** 2 + pi * pd * (2 * binom_Ei + 2 * binom_im1 + (E - i))
        pns_i     = float(np.clip((1 - ps_hat) * num_CE / denom_CE, 0.0, 1.0))

        weights = binom_coeff * (pns_i ** (K_hat - t_arr)) * ((1 - pns_i) ** t_arr)

        for s in range(S):
            mean_rho = np.zeros(4)
            for t in t_arr:
                mean_rho += weights[t] * (
                    (t / K_hat) * 0.25 * np.ones(4) +
                    ((K_hat - t) / K_hat) * ss[s]
                )
            rho_hat[i - 1, s, :] = mean_rho

    return rho_hat


def compute_rho_hat_Em1(ss, pd, pi, ps, E, K_hat_Em1):
    if K_hat_Em1 == 0:
        return np.full((E, ss.shape[0], 4), 0.25)

    S = ss.shape[0]
    rho_hat = np.zeros((E, S, 4), dtype=np.float64)
    ps_hat = 4.0 * ps / 3.0

    t_arr       = np.arange(K_hat_Em1 + 1)
    binom_coeff = np.array([comb(K_hat_Em1, t, exact=False) for t in t_arr])

    for i in range(1, E + 1):
        pns_i = float(np.clip((1 - ps_hat) * (E - i) / E, 0.0, 1.0))
        weights = binom_coeff * (pns_i ** (K_hat_Em1 - t_arr)) * ((1 - pns_i) ** t_arr)

        for s in range(S):
            mean_rho = np.zeros(4)
            for t in t_arr:
                mean_rho += weights[t] * (
                    (t / K_hat_Em1) * 0.25 * np.ones(4) +
                    ((K_hat_Em1 - t) / K_hat_Em1) * ss[s]
                )
            rho_hat[i - 1, s, :] = mean_rho

    return rho_hat


def compute_rho_hat_Ep1(ss, pd, pi, ps, E, K_hat_Ep1):
    if K_hat_Ep1 == 0:
        return np.full((E, ss.shape[0], 4), 0.25)

    S = ss.shape[0]
    rho_hat = np.zeros((E, S, 4), dtype=np.float64)
    ps_hat = 4.0 * ps / 3.0

    t_arr       = np.arange(K_hat_Ep1 + 1)
    binom_coeff = np.array([comb(K_hat_Ep1, t, exact=False) for t in t_arr])

    for i in range(1, E + 1):
        pns_i = float(np.clip((1 - ps_hat) * (E - i + 1) / E, 0.0, 1.0))
        weights = binom_coeff * (pns_i ** (K_hat_Ep1 - t_arr)) * ((1 - pns_i) ** t_arr)

        for s in range(S):
            mean_rho = np.zeros(4)
            for t in t_arr:
                mean_rho += weights[t] * (
                    (t / K_hat_Ep1) * 0.25 * np.ones(4) +
                    ((K_hat_Ep1 - t) / K_hat_Ep1) * ss[s]
                )
            rho_hat[i - 1, s, :] = mean_rho

    return rho_hat


def llr_from_counts_and_rho(X_counts, rho_hat, eps=1e-12):
    log_rho = np.log(rho_hat.clip(min=eps))
    ll = (X_counts[:, np.newaxis, :] * log_rho).sum(-1)

    llr = np.zeros((X_counts.shape[0], 3), dtype=np.float64)
    for ell in range(3):
        idx0 = np.where(u_bits_np[:, ell] == 0)[0]
        idx1 = np.where(u_bits_np[:, ell] == 1)[0]
        ll0  = ll[:, idx0]
        ll1  = ll[:, idx1]
        a0   = ll0.max(axis=1, keepdims=True)
        a1   = ll1.max(axis=1, keepdims=True)
        lse0 = a0[:, 0] + np.log(np.exp(ll0 - a0).sum(axis=1))
        lse1 = a1[:, 0] + np.log(np.exp(ll1 - a1).sum(axis=1))
        llr[:, ell] = lse0 - lse1

    return llr


def bits_to_int_left_msb(bit_matrix):
    L = bit_matrix.shape[1]
    weights = (1 << np.arange(L - 1, -1, -1, dtype=np.int64))
    return (bit_matrix.astype(np.int64) * weights).sum(axis=1)


def ldpc_encode_systematic(x, Gsys):
    return (x @ Gsys) & 1


def sample_composite_channel(d_sym, ss, K, rng):
    n = d_sym.size
    d_obs = np.zeros((n, 4), dtype=np.int32)
    for i in range(n):
        d_obs[i, :] = rng.multinomial(K, ss[d_sym[i], :])
    return d_obs


def sample_without_replacement_from_counts(d_obs, K, rng):
    n = d_obs.shape[0]
    d_sampled = np.zeros((n, K), dtype=np.int16)
    for t in range(n):
        pool = np.repeat(np.arange(1, 5, dtype=np.int16), d_obs[t, :])
        rng.shuffle(pool)
        d_sampled[t, :] = pool[:K]
    return d_sampled


def apply_ids_channel_per_trace(d_sampled, pd, pi, ps_prime, rng):
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


def counts_per_position(d_ID):
    n, _ = d_ID.shape
    out = np.zeros((n, 4), dtype=np.int32)
    for t in range(n):
        bc = np.bincount(d_ID[t, :], minlength=5)
        out[t, :] = bc[1:5]
    return out


def simulate_one_codeword(Gsys, L, K_reads, pd, pi, ps, rng, ss, E=160):
    k = Gsys.shape[0]
    x_info = rng.integers(0, 2, size=k, dtype=np.uint8)
    d = ldpc_encode_systematic(x_info, Gsys.astype(np.uint8))
    rem = d.size % L
    if rem != 0:
        d = np.concatenate([d, np.zeros(L - rem, dtype=d.dtype)])

    d_matrix = d.reshape(-1, L)
    d_sym    = bits_to_int_left_msb(d_matrix)
    T        = d_sym.size

    d_obs     = sample_composite_channel(d_sym.astype(np.int64), ss, K_reads, rng)
    d_sampled = sample_without_replacement_from_counts(d_obs, K_reads, rng)

    ps_prime = (1.0 - pd - pi) * ps
    d_IDall  = apply_ids_channel_per_trace(d_sampled, pd, pi, ps_prime, rng)

    d_ID   = np.zeros((T,     K_reads), dtype=np.int16)
    d_Em1  = np.zeros((T - 1, K_reads), dtype=np.int16)
    d_Ep1  = np.zeros((T,     K_reads), dtype=np.int16)
    n_hat_E = 0; n_hat_Em1 = 0; n_hat_Ep1 = 0

    for kx in range(K_reads):
        is_short = d_IDall[T - 1, kx] == 0
        is_long  = d_IDall[T,     kx] > 0

        if not is_short and not is_long:
            d_ID[:, kx] = d_IDall[:T, kx]
            n_hat_E += 1
        else:
            if is_short:
                nz = np.nonzero(d_IDall[:T, kx])[0]
                strand_len = 0 if nz.size == 0 else int(nz[-1] + 1)
                if strand_len == T - 1:
                    d_Em1[:, kx] = d_IDall[:T - 1, kx]
                    n_hat_Em1 += 1
            elif is_long:
                nz = np.nonzero(d_IDall[:T + 2, kx])[0]
                strand_len = 0 if nz.size == 0 else int(nz[-1] + 1)
                if strand_len == T + 1:
                    d_Ep1[:, kx] = d_IDall[:T, kx]
                    n_hat_Ep1 += 1

    X_E   = counts_per_position(d_ID).astype(np.float64)
    X_Em1 = counts_per_position(d_Em1).astype(np.float64)
    X_Ep1 = counts_per_position(d_Ep1).astype(np.float64)

    X_Em1_padded = np.zeros((T, 4), dtype=np.float64)
    X_Em1_padded[:T - 1, :] = X_Em1

    total = n_hat_E + n_hat_Em1 + n_hat_Ep1
    if total == 0:
        return np.zeros(T * L, dtype=np.float64), x_info

    rho_hat_E   = compute_rho_hat_E(ss, pd, pi, ps, T, n_hat_E)     if n_hat_E   > 0 else None
    rho_hat_Em1 = compute_rho_hat_Em1(ss, pd, pi, ps, T, n_hat_Em1) if n_hat_Em1 > 0 else None
    rho_hat_Ep1 = compute_rho_hat_Ep1(ss, pd, pi, ps, T, n_hat_Ep1) if n_hat_Ep1 > 0 else None

    llr_mat = np.zeros((T, 3), dtype=np.float64)
    if rho_hat_E is not None:
        llr_mat += llr_from_counts_and_rho(X_E,          rho_hat_E)
    if rho_hat_Em1 is not None:
        llr_mat += llr_from_counts_and_rho(X_Em1_padded, rho_hat_Em1)
    if rho_hat_Ep1 is not None:
        llr_mat += llr_from_counts_and_rho(X_Ep1,        rho_hat_Ep1)

    llr_flat = llr_mat.reshape(-1)
    return llr_flat, x_info


def ldpc_decode_logspa(H, llr, max_iters=50, llr_clip=50.0, early_stop=True):
    H = (np.asarray(H).astype(np.int8) & 1)
    m, n = H.shape
    llr = np.asarray(llr, dtype=np.float64)
    single = (llr.ndim == 1)
    if single:
        llr = llr[None, :]
    B, n2 = llr.shape
    if n2 != n:
        raise ValueError(f"LLR length {n2} != H columns {n}.")
    check_nbrs = [np.where(H[i] == 1)[0] for i in range(m)]
    q = [np.clip(llr[:, check_nbrs[i]], -llr_clip, llr_clip).copy() for i in range(m)]
    r = [np.zeros_like(q[i]) for i in range(m)]

    def hard_from_llr(L_):
        return (L_ < 0).astype(np.int8)

    def syndrome_bits(xb):
        return (xb @ H.T) & 1

    ATANH_CLIP = 1.0 - 1e-12
    L_post = llr.copy()
    success = np.zeros(B, dtype=bool)
    for it in range(1, max_iters + 1):
        for i in range(m):
            qs = np.clip(q[i], -llr_clip, llr_clip)
            t  = np.tanh(qs / 2.0)
            prod_all = np.prod(t, axis=1, keepdims=True)
            deg = t.shape[1]
            for k in range(deg):
                denom = t[:, [k]]
                mask  = (np.abs(denom[:, 0]) < 1e-300)
                prod_excl = np.empty_like(denom)
                not_mask  = ~mask
                prod_excl[not_mask, 0] = prod_all[not_mask, 0] / denom[not_mask, 0]
                if np.any(mask):
                    if deg == 1:
                        prod_excl[mask, 0] = 1.0
                    else:
                        t_ex = np.delete(t, k, axis=1)
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
        syn   = syndrome_bits(x_hat)
        ok    = np.all(syn == 0, axis=1)
        success = ok
        if early_stop and np.all(ok):
            break
    x_hat = hard_from_llr(L_post)
    if single:
        return x_hat[0], L_post[0], bool(success[0])
    return x_hat, L_post, success


def _default_iter_total(pd, pi, ps):
    pmax = max(pd, pi, ps)
    return 20000
    if pmax <= 0.0015:
        return 100000
    elif pmax < 0.00275:
        return 50000
    elif pmax < 0.0045:
        return 5000
    else:
        return 1000


def run_bler_point(H, Gsys, L, K_reads, E, pd, pi, ps,
                    max_ldpc_iters, iter_total, batch_size, rng):
    n_codebits = H.shape[1]
    k_info     = Gsys.shape[0]

    n_blocks   = (iter_total + batch_size - 1) // batch_size
    err_blocks = 0
    done       = 0

    for _ in range(n_blocks):
        B = min(batch_size, iter_total - done)
        llr_batch = np.zeros((B, n_codebits), dtype=np.float64)
        Ub        = np.zeros((B, k_info),     dtype=np.uint8)

        for i in range(B):
            llr_flat, u = simulate_one_codeword(
                Gsys=Gsys, L=L, K_reads=K_reads,
                pd=float(pd), pi=float(pi), ps=float(ps),
                rng=rng, ss=ss_np, E=E
            )
            if llr_flat.size >= n_codebits:
                llr_batch[i, :] = llr_flat[:n_codebits]
            else:
                llr_batch[i, :llr_flat.size] = llr_flat
            Ub[i, :] = u

        x_hat_code, _, success = ldpc_decode_logspa(H, llr_batch, max_iters=max_ldpc_iters)
        x_hat_info = x_hat_code[:, :k_info].astype(np.int16)
        u_true     = Ub.astype(np.int16)
        err_blocks += int(np.sum(np.any(x_hat_info != u_true, axis=1)))
        done += B

    return err_blocks / iter_total


def estimate_bler_sweep_n(
    H_path, H_key, Gsys, n_list, pd, pi, ps,
    L=3, E=160,
    max_ldpc_iters=20,
    iter_total_map=None,
    batch_size=2048,
    seed=None,
    save_results_xlsx=None,
):
    """
    Option 1: fixed pd, pi, ps -- sweep number of samples (K_reads).
    pd, pi, ps can each be set independently (e.g. pi=pd=0 for substitution-only,
    ps=0 for ID-only, etc).
    """
    rng = np.random.default_rng(seed)
    A   = loadmat(H_path, squeeze_me=True, struct_as_record=False)
    H   = A[H_key]

    results = {}
    n_listx, bler_listx = [], []

    for K_reads in n_list:
        if iter_total_map is not None and K_reads in iter_total_map:
            iter_total = int(iter_total_map[K_reads])
        else:
            iter_total = _default_iter_total(pd, pi, ps)

        t0 = time.time()
        bler = run_bler_point(
            H, Gsys, L, K_reads, E, pd, pi, ps,
            max_ldpc_iters, iter_total, batch_size, rng
        )
        results[int(K_reads)] = float(bler)
        dt = time.time() - t0
        print(f"n={K_reads:3d}  pd={pd:.5f} pi={pi:.5f} ps={ps:.5f}  "
              f"iter_total={iter_total}  BLER={bler:.6g}  time={dt/60:.2f} min")

        if save_results_xlsx is not None:
            n_listx.append(int(K_reads))
            bler_listx.append(float(bler))
            df = panda.DataFrame({"n": n_listx, "BLER": bler_listx})
            df.to_excel(save_results_xlsx, index=False)
            print(f"  Saved to {save_results_xlsx}")

    return results


def estimate_bler_sweep_prob(
    H_path, H_key, Gsys, prob_list, K_reads,
    L=3, E=160,
    max_ldpc_iters=20,
    iter_total_map=None,
    batch_size=2048,
    seed=None,
    save_results_xlsx=None,
):
    """
    Option 2: fixed number of samples K_reads -- sweep error probabilities.
    prob_list is a list of (pd, pi, ps) tuples, so pd, pi, ps need not be equal
    (e.g. substitution-only: pi=pd=0; ID-only: ps=0; full IDS: all nonzero).
    """
    rng = np.random.default_rng(seed)
    A   = loadmat(H_path, squeeze_me=True, struct_as_record=False)
    H   = A[H_key]

    results = {}
    pd_listx, pi_listx, ps_listx, bler_listx = [], [], [], []

    for (pd, pi, ps) in prob_list:
        pd, pi, ps = float(pd), float(pi), float(ps)
        key = (pd, pi, ps)

        if iter_total_map is not None and key in iter_total_map:
            iter_total = int(iter_total_map[key])
        else:
            iter_total = _default_iter_total(pd, pi, ps)

        t0 = time.time()
        bler = run_bler_point(
            H, Gsys, L, K_reads, E, pd, pi, ps,
            max_ldpc_iters, iter_total, batch_size, rng
        )
        results[key] = float(bler)
        dt = time.time() - t0
        print(f"pd={pd:.5f} pi={pi:.5f} ps={ps:.5f}  n={K_reads}  "
              f"iter_total={iter_total}  BLER={bler:.6g}  time={dt/60:.2f} min")

        if save_results_xlsx is not None:
            pd_listx.append(pd); pi_listx.append(pi); ps_listx.append(ps)
            bler_listx.append(float(bler))
            df = panda.DataFrame({"pd": pd_listx, "pi": pi_listx, "ps": ps_listx, "BLER": bler_listx})
            df.to_excel(save_results_xlsx, index=False)
            print(f"  Saved to {save_results_xlsx}")

    return results


if __name__ == "__main__":
    L = 3
    E = 160

    H_path = "Hmatrices.mat"
    H_key  = "H"
    Gsys   = loadmat(H_path)["Gsys"].astype(np.uint8)

    n_list = list(range(1, 16))
    ps_list = [0.4]
 
    results_sweep_n = {}
    for ps in ps_list:
        ps_tag = str(ps).replace(".", "p")
        results_sweep_n[ps] = estimate_bler_sweep_n(
            H_path=H_path, H_key=H_key, Gsys=Gsys,
            n_list=n_list,
            pd=0.0, pi=0.0, ps=ps,
            L=L, E=E,
            max_ldpc_iters=20, batch_size=2048, seed=42,
            save_results_xlsx=f"BLER_sweep_n_subs_ps{ps_tag}-corrected-30JUN26.xlsx",
        )

    n_list = list(range(1, 16))
    ps_list = [0.001, 0.002]
 
    results_sweep_n = {}
    for psx in ps_list:
        ps_tag = str(psx).replace(".", "p")
        results_sweep_n[psx] = estimate_bler_sweep_n(
            H_path=H_path, H_key=H_key, Gsys=Gsys,
            n_list=n_list,
            pd=psx, pi=psx, ps=0,
            L=L, E=E,
            max_ldpc_iters=20, batch_size=2048, seed=42,
            save_results_xlsx=f"BLER_sweep_n_ID{ps_tag}-corrected-30JUN26.xlsx",
        )

    n_list = list(range(1, 16))
    ps_list = [0.001, 0.002]
 
    results_sweep_n = {}
    for psx in ps_list:
        ps_tag = str(psx).replace(".", "p")
        results_sweep_n[psx] = estimate_bler_sweep_n(
            H_path=H_path, H_key=H_key, Gsys=Gsys,
            n_list=n_list,
            pd=psx, pi=psx, ps=psx,
            L=L, E=E,
            max_ldpc_iters=20, batch_size=2048, seed=42,
            save_results_xlsx=f"BLER_sweep_n_IDS{ps_tag}-corrected-30JUN26.xlsx",
        )


