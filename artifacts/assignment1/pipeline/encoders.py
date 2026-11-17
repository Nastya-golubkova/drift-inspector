"""Frozen encoders: SPECTER2, SemCSE, Qwen3-Embedding."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from . import config


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


def _save_meta(path: Path, meta: dict):
    path.write_text(json.dumps(meta, indent=2))


def embed_specter2(texts: list[str], cache: Path, batch_size: int = 32,
                   normalize: bool = False) -> np.ndarray:
    cache = Path(cache)
    meta_path = cache.with_suffix(".meta.json")
    if cache.exists():
        E = np.load(cache)
        if len(E) == len(texts):
            print(f"cache hit {cache.name} {E.shape}")
            return E
        print(f"cache length mismatch {cache.name}; recomputing")

    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    tok = AutoTokenizer.from_pretrained(config.SPECTER2_MODEL)
    model = AutoAdapterModel.from_pretrained(config.SPECTER2_MODEL)
    model.load_adapter(config.SPECTER2_ADAPTER, source="hf",
                       load_as="proximity", set_active=True)
    model.set_active_adapters("proximity")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=cache.stem):
            enc = tok(
                texts[i:i + batch_size], padding=True, truncation=True,
                max_length=512, return_tensors="pt", return_token_type_ids=False,
            ).to(device)
            out.append(model(**enc).last_hidden_state[:, 0, :].cpu().numpy())
    E = np.vstack(out).astype(np.float32)
    if normalize:
        E = _l2(E)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, E)
    _save_meta(meta_path, {
        "model": config.SPECTER2_MODEL,
        "adapter": config.SPECTER2_ADAPTER,
        "pooling": "CLS",
        "max_length": 512,
        "normalize": normalize,
        "n": len(texts),
        "dim": int(E.shape[1]),
    })
    print(f"saved {cache.name} {E.shape}")
    return E


def embed_semcse(texts: list[str], cache: Path, batch_size: int = 32) -> np.ndarray:
    cache = Path(cache)
    meta_path = cache.with_suffix(".meta.json")
    if cache.exists():
        E = np.load(cache)
        if len(E) == len(texts):
            print(f"cache hit {cache.name} {E.shape}")
            return E

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.SEMCSE_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    E = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, E)
    _save_meta(meta_path, {
        "model": config.SEMCSE_MODEL,
        "library": "sentence-transformers",
        "normalize": True,
        "n": len(texts),
        "dim": int(E.shape[1]),
    })
    print(f"saved {cache.name} {E.shape}")
    return E


def embed_qwen(texts: list[str], cache: Path, batch_size: int = 16,
               instruction: str | None = None) -> np.ndarray:
    """Qwen3-Embedding-0.6B with fixed instruction prefix."""
    cache = Path(cache)
    meta_path = cache.with_suffix(".meta.json")
    instruction = instruction or config.QWEN_INSTRUCTION
    if cache.exists():
        E = np.load(cache)
        if len(E) == len(texts):
            print(f"cache hit {cache.name} {E.shape}")
            return E

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        config.QWEN_MODEL,
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    # Official pattern: instruct + query text for retrieval; for document
    # embedding of abstracts we apply the fixed instruction uniformly.
    prompted = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    E = model.encode(
        prompted, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, E)
    _save_meta(meta_path, {
        "model": config.QWEN_MODEL,
        "instruction": instruction,
        "normalize": True,
        "n": len(texts),
        "dim": int(E.shape[1]),
    })
    print(f"saved {cache.name} {E.shape}")
    return E
