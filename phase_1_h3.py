import os
import json
import time
import h5py
import faiss
import random
import numpy as np
import openslide
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.geometry import shape
from shapely.ops import unary_union

# ============================================================
# PATHS
# ============================================================

FEATURE_PATHS = {

    "CONCH":
    "/home/woody/iwi5/iwi5411h/BIMAP/data/patches_PT_484/conch/20x_256px_128px_overlap/features_conch_v15/PrimaryTumor_HE_484.h5",

    "UNI2":
    "/home/woody/iwi5/iwi5411h/BIMAP/data/patches_PT_484/uni2/20x_256px_128px_overlap/features_uni_v2/PrimaryTumor_HE_484.h5",

    "VIRCHOW":
    "/home/woody/iwi5/iwi5411h/BIMAP/data/patches_PT_484/virchow/20x_256px_128px_overlap/features_virchow/PrimaryTumor_HE_484.h5"

}

WSI_PATH = "/home/woody/iwi5/iwi5411h/BIMAP/data/WSI_PrimaryTumor_CUP/PrimaryTumor_HE_484.svs"

GEOJSON_PATH = "/home/woody/iwi5/iwi5411h/BIMAP/data/WSI_PrimaryTumor_Annotations/PrimaryTumor_HE_484.geojson"

RESULTS_DIR = "/home/woody/iwi5/iwi5411h/BIMAP/results/comparison/foundation_models"

os.makedirs(RESULTS_DIR, exist_ok=True)

PATCH_SIZE = 256
TOP_K = 15
NUM_QUERIES = 30

# ============================================================
# LOAD GEOJSON
# ============================================================

print("\nLoading GeoJSON...")

with open(GEOJSON_PATH, "r") as f:
    geo = json.load(f)

polygons = []

for feature in geo["features"]:

    geom = shape(feature["geometry"])
    polygons.append(geom)

tumor_region = unary_union(polygons)

print("Tumor polygons loaded")

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

def precision_at_k(retrieved_indices, labels, k):

    retrieved = retrieved_indices[:k]

    relevant = np.sum(labels[retrieved] == 1)

    return relevant / k


def average_precision(retrieved_indices, labels):

    precisions = []

    relevant_count = 0

    for rank, idx in enumerate(retrieved_indices, start=1):

        if labels[idx] == 1:

            relevant_count += 1

            precisions.append(
                relevant_count / rank
            )

    if len(precisions) == 0:
        return 0

    return np.mean(precisions)

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

        nlist = 100

        quantizer = faiss.IndexFlatIP(d)

        self.index = faiss.IndexIVFFlat(
            quantizer,
            d,
            nlist,
            faiss.METRIC_INNER_PRODUCT
        )

        self.index.train(X)

        self.index.add(X)

        self.index.nprobe = 10

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
    (len(FEATURE_PATHS), len(RETRIEVAL_METHODS))
)

# ============================================================
# MAIN LOOP
# ============================================================

for model_idx, (model_name, h5_path) in enumerate(FEATURE_PATHS.items()):

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

        center_x = x + PATCH_SIZE / 2
        center_y = y + PATCH_SIZE / 2

        pt = Point(center_x, center_y)

        inside = tumor_region.contains(pt)

        labels.append(int(inside))

    labels = np.array(labels)

    tumor_indices = np.where(labels == 1)[0]

    print("Tumor patches:", len(tumor_indices))

    # --------------------------------------------------------
    # RANDOM TUMOR QUERIES
    # --------------------------------------------------------

    random.seed(42)
    np.random.seed(42)

    query_indices = np.random.choice(
        tumor_indices,
        min(NUM_QUERIES, len(tumor_indices)),
        replace=False
    )

    # --------------------------------------------------------
    # INITIALIZE RETRIEVAL METHODS
    # --------------------------------------------------------

    methods = {

        "BruteForce": BruteForce(features),

        "FAISSFlat": FAISSFlat(features),

        "FAISSIVF": FAISSIVF(features),

        "FAISSHNSW": FAISSHNSW(features)

    }

    all_results[model_name] = {}

    # --------------------------------------------------------
    # TEST RETRIEVAL METHODS
    # --------------------------------------------------------

    for method_idx, (method_name, method) in enumerate(methods.items()):

        print(f"\nRunning {method_name}")

        p5_scores = []
        p10_scores = []
        map_scores = []
        latency_scores = []

        # ----------------------------------------------------
        # MULTIPLE QUERY EVALUATION
        # ----------------------------------------------------

        for query_idx in query_indices:

            query_embedding = features[query_idx]

            start = time.time()

            retrieved_indices = method.search(
                query_embedding,
                TOP_K + 1
            )

            latency = (time.time() - start) * 1000

            retrieved_indices = retrieved_indices[
                retrieved_indices != query_idx
            ][:TOP_K]

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

        # ----------------------------------------------------
        # AVERAGE METRICS
        # ----------------------------------------------------

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

        heatmap_matrix[model_idx, method_idx] = avg_map

# ============================================================
# BARPLOT : BEST mAP PER MODEL
# ============================================================

best_maps = []
model_names = []

for model_name in FEATURE_PATHS.keys():

    maps = [

        all_results[model_name][method]["mAP"]
        for method in RETRIEVAL_METHODS

    ]

    best_maps.append(max(maps))

    model_names.append(model_name)

plt.figure(figsize=(8,5))

plt.bar(
    model_names,
    best_maps
)

for i, v in enumerate(best_maps):

    plt.text(
        i,
        v,
        f"{v:.3f}",
        ha="center"
    )

plt.ylabel("Best mAP")
plt.xlabel("Foundation Model")

plt.title(
    "Foundation Model Comparison"
)

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "foundation_model_map_comparison.png"
    ),

    dpi=300

)

plt.close()

# ============================================================
# HEATMAP : MODEL vs RETRIEVAL
# ============================================================

plt.figure(figsize=(10,6))

im = plt.imshow(
    heatmap_matrix,
    aspect='auto'
)

plt.colorbar(im)

plt.xticks(
    np.arange(len(RETRIEVAL_METHODS)),
    RETRIEVAL_METHODS
)

plt.yticks(
    np.arange(len(model_names)),
    model_names
)

# values inside cells

for i in range(len(model_names)):

    for j in range(len(RETRIEVAL_METHODS)):

        plt.text(
            j,
            i,
            f"{heatmap_matrix[i,j]:.3f}",
            ha="center",
            va="center",
            color="white"
        )

plt.xlabel("Retrieval Algorithm")
plt.ylabel("Foundation Model")

plt.title(
    "mAP Heatmap"
)

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "model_vs_retrieval_heatmap.png"
    ),

    dpi=300

)

plt.close()

# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "foundation_model_summary.txt"
)

with open(summary_path, "w") as f:

    for model_name, vals in all_results.items():

        f.write("\n================================================\n")
        f.write(f"{model_name}\n")
        f.write("================================================\n")

        for method_name, metrics in vals.items():

            f.write(f"\n{method_name}\n")

            for k, v in metrics.items():

                f.write(f"{k}: {v:.4f}\n")

# ============================================================
# FINAL PRINT
# ============================================================

print("\n================================================")
print("EXPERIMENT FINISHED")
print("================================================")

print("\nResults saved to:")
print(RESULTS_DIR)