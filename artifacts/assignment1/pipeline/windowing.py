"""Sentence-aware abstract windowing."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def approx_token_count(text: str) -> int:
    """Whitespace token count as a stable proxy (encoder-agnostic packing)."""
    return len(str(text).split())


@dataclass
class Window:
    paper_id: str
    window_index: int
    start_sent: int
    end_sent: int
    token_count: int
    text: str


def pack_windows(paper_id: str, abstract: str, max_tokens: int = 80) -> list[Window]:
    sents = split_sentences(abstract)
    if not sents:
        return [Window(paper_id, 0, 0, 0, 0, "")]

    windows: list[Window] = []
    i = 0
    while i < len(sents):
        buf: list[str] = []
        toks = 0
        start = i
        while i < len(sents):
            st = approx_token_count(sents[i])
            # Always take at least one sentence even if it exceeds max_tokens.
            if buf and toks + st > max_tokens:
                break
            buf.append(sents[i])
            toks += st
            i += 1
            if not buf:
                break
        text = " ".join(buf)
        windows.append(Window(
            paper_id=paper_id,
            window_index=len(windows),
            start_sent=start,
            end_sent=i - 1,
            token_count=approx_token_count(text),
            text=text,
        ))
    return windows


def build_window_table(papers: pd.DataFrame, max_tokens: int = 80) -> pd.DataFrame:
    rows = []
    for _, r in papers.iterrows():
        for w in pack_windows(str(r["paper_id"]), str(r["abstract"]), max_tokens):
            rows.append({
                "paper_id": w.paper_id,
                "window_index": w.window_index,
                "start_sent": w.start_sent,
                "end_sent": w.end_sent,
                "token_count": w.token_count,
                "text": w.text,
            })
    return pd.DataFrame(rows)


def window_stats(win_df: pd.DataFrame) -> dict:
    per = win_df.groupby("paper_id").size()
    return {
        "n_papers": int(per.shape[0]),
        "n_windows": int(len(win_df)),
        "papers_1_window": int((per == 1).sum()),
        "papers_2_windows": int((per == 2).sum()),
        "papers_3plus_windows": int((per >= 3).sum()),
        "pct_single_window": float((per == 1).mean()),
        "tokens_mean": float(win_df["token_count"].mean()),
        "tokens_median": float(win_df["token_count"].median()),
        "tokens_max": int(win_df["token_count"].max()),
        "windows_per_paper_mean": float(per.mean()),
    }
