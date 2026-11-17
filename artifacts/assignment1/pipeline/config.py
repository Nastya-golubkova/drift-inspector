"""Fixed configuration for Assignment 1 abstract-window experiments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "assignment1"
CACHE = OUT / "caches"
EMB_DIR = CACHE / "embeddings"
WIN_DIR = CACHE / "windows"
SIM_DIR = CACHE / "similarity"
CLUST_DIR = OUT / "clusters"
RES_DIR = OUT / "results"

PAPERS_CSV = ROOT / "data" / "claims_sources" / "emnlp_full" / "papers.csv"
EVAL653_CSV = ROOT / "artifacts" / "extension_full656" / "author_table2_eval_papers.csv"
STOP_CSV = ROOT / "data" / "external" / "STop_topic_classification_dataset_for_scientific_papers.csv"
BASELINE_SPECTER2 = ROOT / "artifacts" / "baseline" / "specter2_abstract_embeddings.npy"
T2_CSV = ROOT / "artifacts" / "baseline" / "stop_alignment_table.csv"

RANDOM_STATE = 42
UMAP_KW = dict(
    n_neighbors=40, n_components=5, min_dist=0.0,
    metric="cosine", random_state=RANDOM_STATE,
)
HDBSCAN_KW = dict(
    min_cluster_size=25, min_samples=5,
    metric="euclidean", cluster_selection_method="eom",
)

SPECTER2_MODEL = "allenai/specter2_aug2023refresh_base"
SPECTER2_ADAPTER = "allenai/specter2_aug2023refresh"
SEMCSE_MODEL = "CLAUSE-Bielefeld/SemCSE_cosine"
QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_INSTRUCTION = (
    "Represent this scientific research abstract by its research topic, "
    "problem, methods, and main contribution."
)

PRIMARY_MAX_WINDOW_TOKENS = 80
ABLATION_WINDOW_TOKENS = (64, 96)
EMBED_BATCH = 32
PAIRF1_PAIRS = 200_000
