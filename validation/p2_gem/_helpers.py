"""Shared constants and utilities for the P2 (GEM) claim test suite."""
import json
from pathlib import Path

from rosetta_tools.paths import ROSETTA_MODELS, ROSETTA_RESULTS

# ---------------------------------------------------------------------------
# P2 corpus — 16 models × 17 concepts = 272 ablation pairs (Appendix A)
# ---------------------------------------------------------------------------
P2_MODELS: list[str] = [
    # Pythia scale ladder (EleutherAI) — 70M–12B
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    # GPT-2 (OpenAI)
    "openai-community/gpt2",
    # OPT (Meta)
    "facebook/opt-6.7b",
    # Qwen2.5 (Alibaba) — 0.5B–14B
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    # Mistral (Mistral AI)
    "mistralai/Mistral-7B-v0.3",
    # Gemma-2 (Google)
    "google/gemma-2-9b",
]

P2_CONCEPTS: list[str] = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

# Model parameter counts (millions) for scale-stratified tests
MODEL_PARAMS_M: dict[str, int] = {
    "EleutherAI/pythia-70m":     70,
    "EleutherAI/pythia-160m":    160,
    "EleutherAI/pythia-410m":    410,
    "EleutherAI/pythia-1b":      1_000,
    "EleutherAI/pythia-2.8b":    2_800,
    "EleutherAI/pythia-6.9b":    6_900,
    "EleutherAI/pythia-12b":     12_000,
    "openai-community/gpt2":     117,
    "facebook/opt-6.7b":         6_700,
    "Qwen/Qwen2.5-0.5B":         500,
    "Qwen/Qwen2.5-1.5B":         1_500,
    "Qwen/Qwen2.5-3B":           3_000,
    "Qwen/Qwen2.5-7B":           7_000,
    "Qwen/Qwen2.5-14B":          14_000,
    "mistralai/Mistral-7B-v0.3": 7_000,
    "google/gemma-2-9b":         9_000,
}


def slug(model_id: str) -> str:
    """Convert HuggingFace model ID to filesystem slug (matching extraction output).
    Matches rosetta_tools.gem._model_slug: replace / and - with _, keep dots."""
    return model_id.replace("/", "_").replace("-", "_")


def load_ablation_pair(model_id: str, concept: str) -> dict | None:
    """Load ablation_gem_{concept}.json["comparison"] for one model/concept pair.
    Returns None if file is missing or comparison key absent."""
    path = ROSETTA_MODELS / slug(model_id) / f"ablation_gem_{concept}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("comparison")
    except Exception:
        return None


def load_gem_nodes(model_id: str, concept: str) -> dict | None:
    """Load gem_{concept}.json for one model/concept pair (node-level data).
    Returns None if file is missing."""
    path = ROSETTA_MODELS / slug(model_id) / f"gem_{concept}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# Result file paths
GEM_EEC_CORPUS_FILE = ROSETTA_RESULTS / "gem_eec_corpus.json"
GEM_ADAPTIVE_WIDTH_FILE = ROSETTA_RESULTS / "gem_adaptive_width" / "gem_adaptive_width.json"
