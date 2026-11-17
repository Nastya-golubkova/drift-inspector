"""Early and late fusion of window embeddings."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


def early_uniform_mean(win_df: pd.DataFrame, E_win: np.ndarray,
                       paper_ids: list[str]) -> np.ndarray:
    assert len(win_df) == len(E_win)
    pid_to_idx = {p: i for i, p in enumerate(paper_ids)}
    dim = E_win.shape[1]
    acc = np.zeros((len(paper_ids), dim), dtype=np.float64)
    cnt = np.zeros(len(paper_ids), dtype=np.float64)
    for i, pid in enumerate(win_df["paper_id"].tolist()):
        j = pid_to_idx[pid]
        acc[j] += E_win[i]
        cnt[j] += 1
    cnt = np.maximum(cnt, 1.0)
    return _l2(acc / cnt[:, None])


def early_token_weighted_mean(win_df: pd.DataFrame, E_win: np.ndarray,
                              paper_ids: list[str]) -> np.ndarray:
    assert len(win_df) == len(E_win)
    pid_to_idx = {p: i for i, p in enumerate(paper_ids)}
    dim = E_win.shape[1]
    acc = np.zeros((len(paper_ids), dim), dtype=np.float64)
    wsum = np.zeros(len(paper_ids), dtype=np.float64)
    weights = np.maximum(win_df["token_count"].to_numpy(dtype=np.float64), 1.0)
    for i, pid in enumerate(win_df["paper_id"].tolist()):
        j = pid_to_idx[pid]
        acc[j] += weights[i] * E_win[i]
        wsum[j] += weights[i]
    wsum = np.maximum(wsum, 1.0)
    return _l2(acc / wsum[:, None])


def _paper_window_slices(win_df: pd.DataFrame, paper_ids: list[str]):
    pid_to_idx = {p: i for i, p in enumerate(paper_ids)}
    paper_of_row = win_df["paper_id"].map(pid_to_idx).to_numpy()
    n = len(paper_ids)
    starts = np.full(n, -1, dtype=np.int64)
    ends = np.full(n, -1, dtype=np.int64)
    for r, p in enumerate(paper_of_row):
        if starts[p] < 0:
            starts[p] = r
        ends[p] = r + 1
    for p in range(n):
        if starts[p] < 0:
            starts[p] = 0
            ends[p] = 0
    return starts, ends


def _pad_windows(E: torch.Tensor, starts: np.ndarray, ends: np.ndarray):
    """Pad to (n, max_w, d); invalid slots get -inf for max / ignored via mask."""
    n = len(starts)
    lengths = ends - starts
    max_w = int(max(int(lengths.max()), 1))
    d = E.shape[1]
    padded = E.new_zeros((n, max_w, d))
    mask = torch.zeros((n, max_w), dtype=torch.bool, device=E.device)
    for i in range(n):
        a, b = int(starts[i]), int(ends[i])
        L = b - a
        if L <= 0:
            continue
        padded[i, :L] = E[a:b]
        mask[i, :L] = True
    return padded, mask


def maxsim_symmetric(E_win: np.ndarray, win_df: pd.DataFrame,
                     paper_ids: list[str], row_batch: int = 512) -> np.ndarray:
    """Symmetric MaxSim via padded window tensors on GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    E_np = _l2(np.asarray(E_win, dtype=np.float32))
    starts, ends = _paper_window_slices(win_df, paper_ids)
    n = len(paper_ids)
    E = torch.from_numpy(E_np).to(device)
    padded, mask = _pad_windows(E, starts, ends)  # (n, max_w, d)
    max_w = padded.shape[1]
    d = padded.shape[2]

    # Flatten gallery: (n*max_w, d); invalid positions zero (dot ~0) but masked later
    gallery = padded.reshape(n * max_w, d)
    gallery_mask = mask.reshape(n * max_w)

    M = torch.zeros((n, n), device=device, dtype=torch.float32)
    for i0 in range(0, n, row_batch):
        i1 = min(n, i0 + row_batch)
        Q = padded[i0:i1].reshape(-1, d)          # (bq*max_w, d)
        qmask = mask[i0:i1].reshape(-1)           # valid query windows
        sim = Q @ gallery.T                       # (bq*mw, n*mw)
        # invalidate invalid gallery cols
        sim[:, ~gallery_mask] = -1e9
        sim = sim.view(i1 - i0, max_w, n, max_w)
        # max over gallery window dim
        sim_max = sim.max(dim=3).values           # (bq, mw, n)
        # mean over valid query windows
        qmask2 = mask[i0:i1].unsqueeze(-1)        # (bq, mw, 1)
        sim_max = sim_max.masked_fill(~qmask2, 0.0)
        denom = mask[i0:i1].sum(dim=1).clamp(min=1).float().unsqueeze(-1)  # (bq,1)
        M[i0:i1] = sim_max.sum(dim=1) / denom
        print(f"  MaxSim directed rows {i1}/{n}", flush=True)

    S = 0.5 * (M + M.T)
    S.fill_diagonal_(1.0)
    S.clamp_(-1.0, 1.0)
    return S.cpu().numpy().astype(np.float32)
