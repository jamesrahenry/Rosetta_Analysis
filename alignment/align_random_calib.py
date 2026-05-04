#!/usr/bin/env python3
"""
align_random_calib.py — Same-prompts confound null for PRH alignment.

Tests whether high Procrustes alignment (0.9807) in the primary PRH result
is driven by shared input-induced covariance structure rather than genuine
representational convergence.

Method:
  For each same-dim model pair:
    1. Extract activations from 200 semantically-neutral random sentences
       at each concept's peak layer (same layer as primary DOM vectors).
    2. Fit Procrustes R on these random-prompt activations (NOT the
       concept-specific calibration used in the primary analysis).
    3. Apply R to the concept DOM vectors from the primary extraction.
    4. Measure aligned cosine — compare to primary result (0.9807).

If alignment holds (~0.90+): same-prompts confound does not explain the result.
If alignment collapses: R was fitting shared input-driven structure, not concept geometry.

Output: ~/rosetta_data/results/prh_random_calib_null.json

Written: 2026-05-04 03:15 UTC
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from transformers import AutoTokenizer

from rosetta_tools.gpu_utils import (
    get_device, get_dtype, release_model,
    load_model_with_retry, requires_quantization,
)
from rosetta_tools.extraction import extract_layer_activations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROSETTA_DATA_ROOT = Path.home() / "rosetta_data"
MODELS_ROOT = ROSETTA_DATA_ROOT / "models"
OUT_PATH = ROSETTA_DATA_ROOT / "results" / "prh_random_calib_null.json"

# Concepts from the primary PRH analysis
CONCEPTS = [
    "certainty", "temporal_order", "credibility",
    "causation", "moral_valence", "sentiment", "negation",
]

# ---------------------------------------------------------------------------
# Random corpus — 220 semantically neutral sentences.
# Topics: geography, biology, cooking, sports, mathematics, history,
# everyday life, technology description, weather. Deliberately avoids
# contrastive certainty/doubt, causal connectives, moral framing, sentiment,
# and temporal sequencing language that would activate the 7 concept axes.
# ---------------------------------------------------------------------------
RANDOM_CORPUS = [
    "The capital of Australia is Canberra, not Sydney.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The Amazon River discharges more water than any other river on Earth.",
    "A standard chess board has 64 squares arranged in an 8×8 grid.",
    "Mount Everest stands at approximately 8,849 metres above sea level.",
    "The human body contains roughly 37 trillion cells.",
    "Sourdough bread is leavened using a fermented flour-and-water starter.",
    "Mercury is the closest planet to the Sun in our solar system.",
    "The Eiffel Tower was completed in 1889 as a temporary exhibition structure.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "A haiku traditionally consists of three lines with 5, 7, and 5 syllables.",
    "The Great Wall of China stretches over 21,000 kilometres in total.",
    "Fibonacci numbers appear frequently in the spiral patterns of sunflowers.",
    "The Pacific Ocean covers more area than all of Earth's landmasses combined.",
    "A standard marathon is 42.195 kilometres, or 26.2 miles.",
    "Penguins are found exclusively in the Southern Hemisphere.",
    "The periodic table currently contains 118 confirmed chemical elements.",
    "A kilogram is defined by the Planck constant, not a physical artefact.",
    "The Sahara Desert occupies roughly 9 million square kilometres of Africa.",
    "Bamboo is technically classified as a grass, not a tree.",
    "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
    "The International Space Station orbits at about 400 kilometres altitude.",
    "Honey has an indefinite shelf life due to its low water content and acidity.",
    "The human genome contains approximately 3 billion base pairs.",
    "Tokyo is the most populous metropolitan area in the world.",
    "A standard basketball court measures 28 metres by 15 metres.",
    "The moon's diameter is roughly a quarter of Earth's diameter.",
    "Avocados are botanically classified as a single-seeded berry.",
    "Glass is an amorphous solid, not a slow-flowing liquid.",
    "The wingspan of an albatross can reach up to 3.5 metres.",
    "Copper is one of the few metals that occurs in its native form in nature.",
    "The Dead Sea is approximately 430 metres below sea level.",
    "Most commercial aircraft cruise at altitudes between 9,000 and 12,000 metres.",
    "A square kilometre equals one million square metres.",
    "The Great Barrier Reef is the world's largest coral reef system.",
    "Hydrogen is the most abundant element in the observable universe.",
    "An octopus has three hearts and blue blood.",
    "The Louvre museum in Paris houses more than 35,000 works of art.",
    "Pluto was reclassified as a dwarf planet in 2006.",
    "The human eye can distinguish approximately 10 million different colours.",
    "A standard piano keyboard has 88 keys.",
    "Lake Superior is the largest of the five Great Lakes by surface area.",
    "The wingspan of a Boeing 747 is approximately 68 metres.",
    "Antarctica holds about 70 percent of Earth's fresh water as ice.",
    "The Andes mountain range runs along the western edge of South America.",
    "The average depth of the ocean is about 3,688 metres.",
    "A light-year is the distance light travels in one year, about 9.46 trillion kilometres.",
    "The Colosseum in Rome was completed around 80 AD.",
    "Salt water freezes at a lower temperature than fresh water.",
    "The diameter of a standard football goal is 7.32 metres wide and 2.44 metres tall.",
    "Venus rotates in the opposite direction to most planets.",
    "Platinum is denser than gold.",
    "The human liver performs over 500 distinct biochemical functions.",
    "The tallest tree species in the world is the coast redwood.",
    "A full moon appears about 14 percent larger at perigee than at apogee.",
    "The Mariana Trench reaches a depth of approximately 11 kilometres.",
    "Table salt is a compound of sodium and chloride ions.",
    "The wingspan of the wandering albatross averages around 300 centimetres.",
    "Rubber is derived from the latex sap of the Hevea brasiliensis tree.",
    "The Nile is generally considered the longest river, at about 6,650 kilometres.",
    "A honeybee visits around 50 to 100 flowers during a single foraging trip.",
    "The Milky Way galaxy contains an estimated 100 to 400 billion stars.",
    "The freezing point of ethanol is −114.1 degrees Celsius.",
    "The Suez Canal connects the Mediterranean Sea to the Red Sea.",
    "An adult blue whale can weigh up to 180 metric tonnes.",
    "The Rosetta Stone was discovered in Egypt in 1799.",
    "Granite is an igneous rock composed primarily of quartz, feldspar, and mica.",
    "The cornea is the transparent front layer of the eye.",
    "The wingspan of a monarch butterfly is between 8.6 and 12.4 centimetres.",
    "A standard shipping container is 6 or 12 metres long.",
    "Carbon dioxide makes up about 0.04 percent of Earth's atmosphere.",
    "The Taj Mahal was built between 1631 and 1648.",
    "Dolphins use echolocation to navigate and hunt in murky water.",
    "The surface temperature of the Sun is approximately 5,500 degrees Celsius.",
    "A standard wine bottle holds 750 millilitres.",
    "The brain of an adult human weighs approximately 1.4 kilograms.",
    "The deepest lake in the world is Lake Baikal in Siberia.",
    "Monarch butterflies migrate up to 4,800 kilometres each year.",
    "The chemical symbol for gold is Au, from the Latin word aurum.",
    "The Parthenon in Athens was constructed between 447 and 432 BC.",
    "A standard USB 3.0 port transfers data at up to 5 gigabits per second.",
    "The Atacama Desert in Chile is one of the driest places on Earth.",
    "A square metre of office paper holds roughly 500 sheets.",
    "The wingspan of a wandering albatross can exceed three metres.",
    "The circumference of Earth at the equator is approximately 40,075 kilometres.",
    "Seawater is about 3.5 percent dissolved salts by mass.",
    "The Pythagorean theorem states that a² + b² = c² for right-angled triangles.",
    "Gold has an atomic number of 79.",
    "The Hubble Space Telescope was launched in April 1990.",
    "A standard brick measures roughly 215 × 102.5 × 65 millimetres.",
    "The blue whale is the largest animal known to have ever existed.",
    "Oxygen makes up approximately 21 percent of Earth's atmosphere.",
    "The International Date Line runs roughly along the 180th meridian.",
    "Silicon is the second most abundant element in Earth's crust.",
    "The wingspan of a peregrine falcon is between 74 and 120 centimetres.",
    "Lake Victoria is the largest lake in Africa by surface area.",
    "The Burj Khalifa stands at 828 metres, making it the tallest building.",
    "A standard football pitch is between 100 and 110 metres long.",
    "Obsidian is a naturally occurring volcanic glass.",
    "The human skeleton consists of 206 bones in an adult.",
    "The wavelength of visible light ranges from about 380 to 700 nanometres.",
    "Chile is the world's longest country from north to south.",
    "A standard A4 sheet measures 210 × 297 millimetres.",
    "The cheetah is the fastest land animal, reaching speeds of up to 112 km/h.",
    "The Mississippi River drains about 40 percent of the continental United States.",
    "Diamonds are the hardest natural material, rating 10 on the Mohs scale.",
    "The orbit of the Moon around Earth takes approximately 27.3 days.",
    "The thermal conductivity of copper is about 400 watts per metre-kelvin.",
    "Iceland is the most volcanically active country in the world per unit area.",
    "The wingspan of a bald eagle ranges from 183 to 244 centimetres.",
    "A standard Olympic swimming pool holds 2,500,000 litres of water.",
    "The chemical formula for water is H₂O.",
    "The Gobi Desert spans parts of China and Mongolia.",
    "A standard deck of playing cards contains 52 cards in 4 suits.",
    "The tallest building in the world as of 2024 is the Burj Khalifa.",
    "Stalactites grow down from cave ceilings; stalagmites grow up from floors.",
    "The Coriolis effect deflects moving objects to the right in the Northern Hemisphere.",
    "A kilobyte is 1,024 bytes; a megabyte is 1,024 kilobytes.",
    "The wingspan of an emperor penguin's flippers is around 30 centimetres.",
    "The density of iron at room temperature is approximately 7.87 g/cm³.",
    "The Amazon rainforest produces about 20 percent of the world's oxygen.",
    "A standard tennis court is 23.77 metres long and 10.97 metres wide.",
    "The Hoover Dam spans the Colorado River on the border of Nevada and Arizona.",
    "The boiling point of nitrogen is −195.8 degrees Celsius.",
    "The Sahel is a semi-arid transition zone between the Sahara and the savanna.",
    "The wingspan of a golden eagle ranges from 180 to 234 centimetres.",
    "Concrete is made from cement, water, sand, and aggregate.",
    "The human ear can detect sounds between 20 and 20,000 hertz.",
    "The Drake Passage separates South America from Antarctica.",
    "A standard brick wall uses about 48 bricks per square metre of wall area.",
    "The wingspan of a California condor can reach up to 3 metres.",
    "The melting point of iron is approximately 1,538 degrees Celsius.",
    "The Bering Strait separates Russia from Alaska at its narrowest point.",
    "A standard SD card uses NAND flash memory.",
    "The wingspan of a harpy eagle can reach up to 224 centimetres.",
    "The Himalayas were formed by the collision of the Indian and Eurasian plates.",
    "A standard rugby union pitch is between 94 and 100 metres long.",
    "The speed of sound in air at 20°C is approximately 343 metres per second.",
    "The Strait of Gibraltar connects the Atlantic Ocean to the Mediterranean Sea.",
    "Tungsten has the highest melting point of any element, at 3,422 degrees Celsius.",
    "The wingspan of a barn owl averages about 85 centimetres.",
    "The Panama Canal is 82 kilometres long.",
    "Aluminum is the most abundant metal in Earth's crust.",
    "A standard golf hole is 108 millimetres in diameter.",
    "The wingspan of a swan ranges from 200 to 238 centimetres.",
    "The Ganges River flows through India and Bangladesh.",
    "A standard ship anchor for a large vessel weighs between 10 and 20 tonnes.",
    "The wingspan of a great horned owl is between 91 and 153 centimetres.",
    "The Caspian Sea is the world's largest inland body of water.",
    "A standard soccer ball has a circumference of 68 to 70 centimetres.",
    "The wingspan of a flamingo is between 95 and 100 centimetres.",
    "The Congo River has the second-largest flow by volume of any river.",
    "A standard violin has four strings tuned to G, D, A, and E.",
    "The wingspan of a red-tailed hawk ranges from 114 to 133 centimetres.",
    "The Lena River is one of the longest rivers in the world.",
    "A standard bicycle wheel has between 28 and 36 spokes.",
    "The wingspan of a snowy owl is between 125 and 150 centimetres.",
    "The Baltic Sea has a lower salt content than typical ocean water.",
    "A standard baseball diamond has 90-foot base paths.",
    "The wingspan of a white stork is between 155 and 215 centimetres.",
    "The Columbia River forms most of the border between Oregon and Washington.",
    "A standard guitar has six strings tuned E, A, D, G, B, E.",
    "The wingspan of a black vulture is between 132 and 150 centimetres.",
    "The Yellow River is the second-longest river in China.",
    "A standard stopwatch can measure time to one-hundredth of a second.",
    "The wingspan of a grey heron ranges from 155 to 175 centimetres.",
    "The Tigris and Euphrates rivers flow through the Mesopotamian plain.",
    "A standard car tyre pressure is between 30 and 35 psi.",
    "The wingspan of a pelican can be up to 360 centimetres.",
    "The Ural Mountains form the conventional boundary between Europe and Asia.",
    "A standard microwave oven operates at 2.45 gigahertz.",
    "The wingspan of a turkey vulture ranges from 160 to 183 centimetres.",
    "The Rhine River flows through Switzerland, Germany, and the Netherlands.",
    "A standard commercial airliner has between 100 and 550 passenger seats.",
    "The wingspan of a secretary bird can reach up to 207 centimetres.",
    "The Volga River is the longest river in Europe.",
    "A standard incandescent light bulb converts about 5 percent of energy to light.",
    "The wingspan of a steller's sea eagle can reach up to 250 centimetres.",
    "The Danube River flows through 10 countries in Europe.",
    "A standard swimming pool lap is 50 metres in competition pools.",
    "The wingspan of a common buzzard ranges from 109 to 136 centimetres.",
    "The Murray-Darling river system is the longest in Australia.",
    "A standard football weighs between 410 and 450 grams.",
    "The wingspan of an osprey ranges from 127 to 180 centimetres.",
    "The Zambezi River in Africa flows over Victoria Falls.",
    "A standard basketball weighs between 567 and 624 grams.",
    "The wingspan of a short-eared owl is between 85 and 110 centimetres.",
    "The Rhine delta empties into the North Sea in the Netherlands.",
    "A standard sprinting track is 400 metres per lap.",
    "The wingspan of a peregrine falcon averages about 95 centimetres.",
    "The Niger River flows through Guinea, Mali, Niger, and Nigeria.",
    "A standard cricket pitch is 20.12 metres between wickets.",
    "The wingspan of a lanner falcon is between 95 and 105 centimetres.",
    "The Mekong River flows through China, Myanmar, Laos, Thailand, Cambodia, and Vietnam.",
    "A standard badminton court is 13.4 metres long and 6.1 metres wide.",
    "The wingspan of a common raven is between 115 and 150 centimetres.",
    "The Irrawaddy River is the principal river of Myanmar.",
    "A standard squash court is 9.75 metres long and 6.4 metres wide.",
    "The wingspan of a black kite averages about 140 centimetres.",
    "The Orinoco River drains a large part of Venezuela and Colombia.",
    "A standard handball court is 40 metres long and 20 metres wide.",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def load_model_results(model_slug: str) -> dict:
    """Load all caz_{concept}.json files for a model slug."""
    model_dir = MODELS_ROOT / model_slug
    results = {}
    for concept in CONCEPTS:
        caz_path = model_dir / f"caz_{concept}.json"
        if caz_path.exists():
            with caz_path.open() as f:
                results[concept] = json.load(f)
    return results


def get_model_id_from_slug(slug: str) -> str:
    """Reverse the slug → model_id mapping (underscore → slash for first segment)."""
    # Stored model_id is in the JSON, read it
    for concept in CONCEPTS:
        caz_path = MODELS_ROOT / slug / f"caz_{concept}.json"
        if caz_path.exists():
            with caz_path.open() as f:
                return json.load(f)["model_id"]
    return slug.replace("_", "/", 1)


def discover_same_dim_models() -> dict[int, list[tuple[str, str]]]:
    """Return {hidden_dim: [(model_slug, model_id), ...]} for all extracted models."""
    dim_to_models: dict[int, list[tuple[str, str]]] = {}
    for model_dir in sorted(MODELS_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        # Check at least one concept extracted
        has_any = False
        hidden_dim = None
        model_id = None
        for concept in CONCEPTS:
            caz_path = model_dir / f"caz_{concept}.json"
            if caz_path.exists():
                with caz_path.open() as f:
                    data = json.load(f)
                hidden_dim = data.get("hidden_dim")
                model_id = data.get("model_id")
                has_any = True
                break
        if has_any and hidden_dim and model_id:
            dim_to_models.setdefault(hidden_dim, []).append(
                (model_dir.name, model_id)
            )
    return dim_to_models


def extract_random_activations(
    model_id: str,
    concept_peak_layers: dict[str, int],
    n_random: int = 200,
    device: str = "cuda",
    batch_size: int = 8,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Extract activations at concept peak layers from random corpus sentences.

    Returns {concept: activations[n_random, hidden_dim]} — no pos/neg split,
    just the n_random sentences' activations at that concept's peak layer.
    """
    rng = np.random.default_rng(seed)
    sentences = list(RANDOM_CORPUS)
    rng.shuffle(sentences)
    texts = sentences[:n_random]

    log.info("  Loading %s for random-calibration extraction...", model_id)
    load_4bit = requires_quantization(model_id) == "4bit"
    dtype = get_dtype(device)

    try:
        model = load_model_with_retry(
            model_id,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            load_in_4bit=load_4bit,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        log.error("  Failed to load %s: %s", model_id, e)
        return {}

    concept_activations: dict[str, np.ndarray] = {}
    unique_peak_layers = set(concept_peak_layers.values())

    # Extract all layers at once, then slice per-concept
    log.info("  Extracting random-text activations at %d unique layers...",
             len(unique_peak_layers))
    try:
        # all_layer_acts: list[ndarray[n_texts, hidden_dim]] indexed by layer
        all_layer_acts = extract_layer_activations(
            model, tokenizer, texts,
            device=device, batch_size=batch_size, pool="last",
        )
        # Exclude embedding layer (index 0)
        transformer_acts = all_layer_acts[1:]

        for concept, peak_layer in concept_peak_layers.items():
            if peak_layer < len(transformer_acts):
                concept_activations[concept] = (
                    transformer_acts[peak_layer].astype(np.float32)
                )
            else:
                log.warning("  Peak layer %d out of range for %s/%s",
                            peak_layer, model_id, concept)

    except Exception as e:
        log.error("  Activation extraction failed for %s: %s", model_id, e)
    finally:
        release_model(model)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    log.info("  Extracted random calibration for %d concepts", len(concept_activations))
    return concept_activations


def run_random_calib_null(n_random: int = 200, seed: int = 42) -> dict:
    """Main experiment: fit R on random-text activations, test concept DOM alignment."""
    device = get_device()

    # Discover all same-dim model groups
    dim_to_models = discover_same_dim_models()
    same_dim_groups = {d: ms for d, ms in dim_to_models.items() if len(ms) >= 2}
    log.info("Same-dim groups: %s", {d: len(ms) for d, ms in same_dim_groups.items()})

    # All unique models needed
    all_needed: dict[str, str] = {}  # slug → model_id
    for models in same_dim_groups.values():
        for slug, model_id in models:
            all_needed[slug] = model_id

    # Load concept peak layers and DOM vectors for all needed models
    log.info("Loading peak layers and DOM vectors from existing results...")
    model_peak_layers: dict[str, dict[str, int]] = {}    # slug → concept → layer
    model_dom_vectors: dict[str, dict[str, np.ndarray]] = {}  # slug → concept → vec

    for slug in all_needed:
        results = load_model_results(slug)
        peak_layers = {}
        dom_vecs = {}
        for concept, data in results.items():
            peak_layers[concept] = data["layer_data"]["peak_layer"]
            metrics = data["layer_data"]["metrics"]
            peak_metric = next(
                (m for m in metrics if m["layer"] == peak_layers[concept]), None
            )
            if peak_metric and "dom_vector" in peak_metric:
                dom_vecs[concept] = np.array(peak_metric["dom_vector"], dtype=np.float64)
        model_peak_layers[slug] = peak_layers
        model_dom_vectors[slug] = dom_vecs
        log.info("  %s: %d concepts with peaks+DOM", slug, len(dom_vecs))

    # Extract random-text activations for each needed model
    random_acts: dict[str, dict[str, np.ndarray]] = {}  # slug → concept → acts

    for slug, model_id in all_needed.items():
        peak_layers = model_peak_layers.get(slug, {})
        if not peak_layers:
            log.warning("No peak layers for %s, skipping", slug)
            continue
        acts = extract_random_activations(
            model_id=model_id,
            concept_peak_layers=peak_layers,
            n_random=n_random,
            device=device,
            seed=seed,
        )
        if acts:
            random_acts[slug] = acts

    # Compute Procrustes alignment using random-text calibration
    log.info("Computing Procrustes alignment with random-text calibration...")
    results_by_concept: dict[str, list[float]] = {c: [] for c in CONCEPTS}
    pair_results = []

    for hidden_dim, models in same_dim_groups.items():
        slugs = [s for s, _ in models]
        for i, src_slug in enumerate(slugs):
            for tgt_slug in slugs:
                if src_slug == tgt_slug:
                    continue

                for concept in CONCEPTS:
                    src_rand = random_acts.get(src_slug, {}).get(concept)
                    tgt_rand = random_acts.get(tgt_slug, {}).get(concept)
                    src_dom = model_dom_vectors.get(src_slug, {}).get(concept)
                    tgt_dom = model_dom_vectors.get(tgt_slug, {}).get(concept)

                    if any(x is None for x in [src_rand, tgt_rand, src_dom, tgt_dom]):
                        continue

                    try:
                        src_rand_f = src_rand.astype(np.float64)
                        tgt_rand_f = tgt_rand.astype(np.float64)

                        # Fit R on random-text activations
                        R, _ = orthogonal_procrustes(src_rand_f, tgt_rand_f)

                        # Apply to concept DOM vectors
                        rotated_src_dom = R.T @ src_dom
                        aligned_cos = cosine_similarity(rotated_src_dom, tgt_dom)
                        raw_cos = cosine_similarity(src_dom, tgt_dom)

                        results_by_concept[concept].append(aligned_cos)
                        pair_results.append({
                            "concept": concept,
                            "source": src_slug,
                            "target": tgt_slug,
                            "hidden_dim": hidden_dim,
                            "random_calib_aligned_cosine": round(aligned_cos, 4),
                            "raw_cosine": round(raw_cos, 4),
                        })
                    except Exception as e:
                        log.warning("Procrustes failed %s/%s/%s: %s",
                                    concept, src_slug, tgt_slug, e)

    # Summary
    concept_means = {
        c: round(float(np.mean(v)), 4) if v else None
        for c, v in results_by_concept.items()
    }
    all_vals = [x for vals in results_by_concept.values() for x in vals]
    grand_mean = round(float(np.mean(all_vals)), 4) if all_vals else None

    log.info("\n=== Random-calibration null summary ===")
    log.info("  Grand mean (random-calib R):  %.4f", grand_mean or 0)
    log.info("  Primary result (concept-calib R): 0.9807")
    for concept, mean in sorted(concept_means.items()):
        log.info("    %-20s %.4f", concept, mean or 0)

    output = {
        "experiment": "random_calibration_null",
        "description": (
            "Procrustes R fit on 200 neutral random-text activations at each "
            "concept's peak layer, applied to concept DOM vectors. Tests whether "
            "primary alignment is driven by shared input-induced covariance."
        ),
        "n_random": n_random,
        "seed": seed,
        "primary_result_for_comparison": 0.9807,
        "grand_mean_random_calib": grand_mean,
        "concept_means": concept_means,
        "n_pairs": len(pair_results),
        "pair_results": pair_results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved → %s", OUT_PATH)

    return output


if __name__ == "__main__":
    run_random_calib_null(n_random=200, seed=42)
