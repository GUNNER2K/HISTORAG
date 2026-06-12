import os
import json
import time
import h5py
import faiss
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import openslide

from PIL import Image
from pathlib import Path

from shapely.geometry import Point
from shapely.geometry import shape
from shapely.ops import unary_union

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--demo",
    action="store_true",
    help="Run demo configuration"
)

args = parser.parse_args()

USE_DEMO = args.demo

# ============================================================
# CONFIG
# ============================================================

if USE_DEMO:

    import configs.demo_config as config

else:

    import configs.full_config as config

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

# ============================================================
# RESULTS
# ============================================================

RESULTS_DIR = Path(config.RESULTS_DIR) / "h3"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# ============================================================
# LOAD GEOJSON
# ============================================================

print("\nLoading GeoJSON...")

with open(config.GEOJSON_PATH, "r") as f:

    geo = json.load(f)

polygons = []

for feature in geo["features"]:

    geom = shape(
        feature["geometry"]
    )

    polygons.append(geom)

tumor_region = unary_union(
    polygons
)

print("Tumor polygons loaded")

# ============================================================
# DEMO MODE
# ============================================================

if USE_DEMO:

    print("\n================================================")
    print("RUNNING DEMO MODE")
    print("================================================")

    wsi_img = Image.open(
        config.WSI_SMALL_PATH
    ).convert("RGB")

    wsi_img = np.array(
        wsi_img
    )

# ============================================================
# FULL MODE
# ============================================================

else:

    print("\n================================================")
    print("RUNNING FULL MODE")
    print("================================================")

# ============================================================
# RETRIEVAL METHODS
# ============================================================

RETRIEVAL_METHODS = [

    "BruteForce",
    "FAISSFlat",
    "FAISSIVF",
    "FAISSHNSW"

]

# ============================================================
# METRICS
# ============================================================

def precision_at_k(
    retrieved_indices,
    labels,
    k
):

    retrieved = retrieved_indices[:k]

    relevant = np.sum(
        labels[retrieved] == 1
    )

    return relevant / k


def average_precision(
    retrieved_indices,
    labels
):

    precisions = []

    relevant_count = 0

    for rank, idx in enumerate(
        retrieved_indices,
        start=1
    ):

        if labels[idx] == 1:

            relevant_count += 1

            precisions.append(
                relevant_count / rank
            )

    if len(precisions) == 0:

        return 0

    return np.mean(
        precisions
    )

# ============================================================
# RETRIEVAL CLASSES
# ============================================================

class BruteForce:

    def __init__(self, X):

        self.X = X

    def search(self, q, k):

        sims = self.X @ q

        idx = np.argsort(-sims)[:k]

        return idx


class FAISSFlat:

    def __init__(self, X):

        d = X.shape[1]

        self.index = faiss.IndexFlatIP(d)

        self.index.add(X)

    def search(self, q, k):

        q = q.reshape(1, -1)

        _, idx = self.index.search(q, k)

        return idx[0]


class FAISSIVF:

    def __init__(self, X):

        d = X.shape[1]

        nlist = 10

        quantizer = faiss.IndexFlatIP(d)

        self.index = faiss.IndexIVFFlat(
            quantizer,
            d,
            nlist,
            faiss.METRIC_INNER_PRODUCT
        )

        self.index.train(X)

        self.index.add(X)

        self.index.nprobe = 5

    def search(self, q, k):

        q = q.reshape(1, -1)

        _, idx = self.index.search(q, k)

        return idx[0]


class FAISSHNSW:

    def __init__(self, X):

        d = X.shape[1]

        self.index = faiss.IndexHNSWFlat(d, 32)

        self.index.hnsw.efConstruction = 40

        self.index.add(X)

    def search(self, q, k):

        q = q.reshape(1, -1)

        _, idx = self.index.search(q, k)

        return idx[0]

# ============================================================
# STORAGE
# ============================================================

all_results = {}

heatmap_matrix = np.zeros(

    (
        len(config.FEATURE_PATHS),
        len(RETRIEVAL_METHODS)
    )

)

# ============================================================
# MAIN LOOP
# ============================================================

for model_idx, (

    model_name,
    h5_path

) in enumerate(
    config.FEATURE_PATHS.items()
):

    print("\n================================================")
    print(f"PROCESSING MODEL: {model_name}")
    print("================================================")

    # --------------------------------------------------------
    # LOAD FEATURES
    # --------------------------------------------------------

    with h5py.File(h5_path, "r") as f:

        coords = f["coords"][:]

        features = f["features"][:]

    print("Coords:", coords.shape)

    print("Features:", features.shape)

    # --------------------------------------------------------
    # NORMALIZE FEATURES
    # --------------------------------------------------------

    features = features.astype(np.float32)

    features = features / np.linalg.norm(
        features,
        axis=1,
        keepdims=True
    )

    # --------------------------------------------------------
    # LABEL PATCHES
    # --------------------------------------------------------

    labels = []

    for x, y in coords:

        center_x = x + config.PATCH_SIZE / 2

        center_y = y + config.PATCH_SIZE / 2

        pt = Point(
            center_x,
            center_y
        )

        inside = tumor_region.contains(pt)

        labels.append(
            int(inside)
        )

    labels = np.array(labels)

    tumor_indices = np.where(
        labels == 1
    )[0]

    print(
        "Tumor patches:",
        len(tumor_indices)
    )

    # --------------------------------------------------------
    # PATCH EXTRACTION
    # --------------------------------------------------------

    def get_patch(idx):

        x, y = coords[idx]

        x = int(x)
        y = int(y)

        if USE_DEMO:

            patch = wsi_img[
                y:y + config.PATCH_SIZE,
                x:x + config.PATCH_SIZE
            ]

            return patch

        else:

            return np.zeros(
                (
                    config.PATCH_SIZE,
                    config.PATCH_SIZE,
                    3
                ),
                dtype=np.uint8
            )

    # --------------------------------------------------------
    # QUERY SELECTION
    # --------------------------------------------------------

    query_indices = np.random.choice(

        tumor_indices,

        min(
            config.NUM_QUERIES,
            len(tumor_indices)
        ),

        replace=False

    )

    # --------------------------------------------------------
    # METHODS
    # --------------------------------------------------------

    methods = {

        "BruteForce":
        BruteForce(features),

        "FAISSFlat":
        FAISSFlat(features),

        "FAISSIVF":
        FAISSIVF(features),

        "FAISSHNSW":
        FAISSHNSW(features)

    }

    all_results[model_name] = {}

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    for method_idx, (

        method_name,
        method

    ) in enumerate(methods.items()):

        print(f"\nRunning {method_name}")

        p5_scores = []
        p10_scores = []
        map_scores = []
        latency_scores = []

        for q_num, query_idx in enumerate(query_indices):

            query_embedding = features[query_idx]

            start = time.time()

            retrieved_indices = method.search(

                query_embedding,
                config.TOP_K + 1

            )

            latency = (
                time.time() - start
            ) * 1000

            retrieved_indices = retrieved_indices[
                retrieved_indices != query_idx
            ][:config.TOP_K]

            p5 = precision_at_k(
                retrieved_indices,
                labels,
                5
            )

            p10 = precision_at_k(
                retrieved_indices,
                labels,
                10
            )

            ap = average_precision(
                retrieved_indices,
                labels
            )

            p5_scores.append(p5)
            p10_scores.append(p10)
            map_scores.append(ap)
            latency_scores.append(latency)

        avg_p5 = np.mean(p5_scores)
        avg_p10 = np.mean(p10_scores)
        avg_map = np.mean(map_scores)
        avg_latency = np.mean(latency_scores)

        print(f"P@5     : {avg_p5:.4f}")
        print(f"P@10    : {avg_p10:.4f}")
        print(f"mAP     : {avg_map:.4f}")
        print(f"Latency : {avg_latency:.4f}")

        all_results[model_name][method_name] = {

            "P@5": avg_p5,
            "P@10": avg_p10,
            "mAP": avg_map,
            "latency": avg_latency

        }

        heatmap_matrix[
            model_idx,
            method_idx
        ] = avg_map

print("\n================================================")
print("EXPERIMENT FINISHED")
print("================================================")

print("\nResults saved to:")
print(RESULTS_DIR)