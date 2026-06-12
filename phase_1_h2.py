
import os
import json
import time
import h5py
import faiss
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

from shapely.geometry import Point
from shapely.geometry import shape
from shapely.ops import unary_union

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(42)
np.random.seed(42)

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

H5_PATH = (
    ROOT_DIR
    / "demo_data"
    / "sample_uni.h5"
)

WSI_PATH = (
    ROOT_DIR
    / "demo_data"
    / "small_sample_wsi.jpg"
)

GEOJSON_PATH = (
    ROOT_DIR
    / "demo_data"
    / "sample_annotations.geojson"
)

RESULTS_DIR = (
    ROOT_DIR
    / "results"
    / "comparison"
    / "uni"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

PATCH_SIZE = 256
TOP_K = 15
NUM_QUERIES = 10

# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading H5 file...")

with h5py.File(H5_PATH, "r") as f:

    coords = f["coords"][:]
    features = f["features"][:]

print("Coords shape:", coords.shape)
print("Features shape:", features.shape)

# ============================================================
# NORMALIZE FEATURES
# ============================================================

features = features.astype(np.float32)

features = features / np.linalg.norm(
    features,
    axis=1,
    keepdims=True
)

# ============================================================
# LOAD GEOJSON
# ============================================================

print("\nLoading GeoJSON...")

with open(GEOJSON_PATH, "r") as f:

    geo = json.load(f)

polygons = []

for feature in geo["features"]:

    geom = shape(
        feature["geometry"]
    )

    polygons.append(geom)

tumor_region = unary_union(polygons)

print("Tumor polygons loaded")

# ============================================================
# LOAD DEMO IMAGE
# ============================================================

print("\nLoading demo WSI PNG...")

wsi_img = Image.open(
    WSI_PATH
).convert("RGB")

wsi_img = np.array(wsi_img)

print("Demo image shape:", wsi_img.shape)

# ============================================================
# LABEL PATCHES
# ============================================================

print("\nAssigning labels...")

labels = []

for x, y in coords:

    center_x = x + PATCH_SIZE / 2
    center_y = y + PATCH_SIZE / 2

    pt = Point(
        center_x,
        center_y
    )

    inside = tumor_region.contains(pt)

    labels.append(int(inside))

labels = np.array(labels)

tumor_indices = np.where(
    labels == 1
)[0]

print("Total patches:", len(labels))
print("Tumor patches:", len(tumor_indices))
print("Background patches:", np.sum(labels == 0))

if len(tumor_indices) == 0:

    raise ValueError(
        "No tumor patches found. "
        "Check coordinate alignment."
    )

# ============================================================
# QUERY SELECTION
# ============================================================

query_indices = np.random.choice(

    tumor_indices,

    min(
        NUM_QUERIES,
        len(tumor_indices)
    ),

    replace=False

)

print(f"\nUsing {len(query_indices)} tumor queries")

# ============================================================
# PATCH EXTRACTION
# ============================================================

def get_patch(idx):

    x, y = coords[idx]

    x = int(x)
    y = int(y)

    patch = wsi_img[
        y:y+PATCH_SIZE,
        x:x+PATCH_SIZE
    ]

    return patch

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

    return np.mean(precisions)

# ============================================================
# RETRIEVAL BACKENDS
# ============================================================

class BruteForce:

    def __init__(self, X):

        self.X = X

    def search(self, q, k):

        sims = self.X @ q

        idx = np.argsort(-sims)[:k]

        return idx, sims[idx]


class FAISSFlat:

    def __init__(self, X):

        d = X.shape[1]

        self.index = faiss.IndexFlatIP(d)

        self.index.add(X)

    def search(self, q, k):

        q = q.reshape(1, -1)

        scores, idx = self.index.search(q, k)

        return idx[0], scores[0]


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

        scores, idx = self.index.search(q, k)

        return idx[0], scores[0]


class FAISSHNSW:

    def __init__(self, X):

        d = X.shape[1]

        self.index = faiss.IndexHNSWFlat(
            d,
            32
        )

        self.index.hnsw.efConstruction = 40

        self.index.add(X)

    def search(self, q, k):

        q = q.reshape(1, -1)

        scores, idx = self.index.search(q, k)

        return idx[0], scores[0]

# ============================================================
# INITIALIZE METHODS
# ============================================================

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

# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = {}

for name, method in methods.items():

    print(f"\nRunning {name}")

    p5_scores = []
    p10_scores = []
    map_scores = []
    latency_scores = []

    # --------------------------------------------------------
    # MULTI QUERY EVALUATION
    # --------------------------------------------------------

    for query_idx in query_indices:

        query_embedding = features[query_idx]

        start = time.time()

        retrieved_indices, scores = method.search(
            query_embedding,
            TOP_K + 1
        )

        latency = (
            time.time() - start
        ) * 1000

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

    # --------------------------------------------------------
    # AVERAGE METRICS
    # --------------------------------------------------------

    avg_p5 = np.mean(p5_scores)
    avg_p10 = np.mean(p10_scores)
    avg_map = np.mean(map_scores)
    avg_latency = np.mean(latency_scores)

    results[name] = {

        "P@5": avg_p5,
        "P@10": avg_p10,
        "mAP": avg_map,
        "latency": avg_latency

    }

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    fixed_query = query_indices[0]

    query_embedding = features[fixed_query]

    retrieved_indices, scores = method.search(
        query_embedding,
        TOP_K + 1
    )

    retrieved_indices = retrieved_indices[
        retrieved_indices != fixed_query
    ][:TOP_K]

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(12, 12)
    )

    axes = axes.flatten()

    axes[0].imshow(
        get_patch(fixed_query)
    )

    axes[0].set_title("QUERY")
    axes[0].axis("off")

    for i, idx in enumerate(retrieved_indices):

        axes[i + 1].imshow(
            get_patch(idx)
        )

        label = (
            "Tumor"
            if labels[idx] == 1
            else "BG"
        )

        axes[i + 1].set_title(label)

        axes[i + 1].axis("off")

    for j in range(
        len(retrieved_indices) + 1,
        16
    ):

        axes[j].axis("off")

    plt.suptitle(

        f"{name}\n"
        f"AVG P@5={avg_p5:.2f} | "
        f"AVG P@10={avg_p10:.2f} | "
        f"AVG mAP={avg_map:.2f} | "
        f"AVG Latency={avg_latency:.2f}ms"

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            RESULTS_DIR,
            f"{name}_retrieval.png"
        ),

        dpi=300

    )

    plt.close()

# ============================================================
# SORT METHODS BY LATENCY
# ============================================================

sorted_methods = sorted(

    results.keys(),

    key=lambda x:
    results[x]["latency"],

    reverse=True

)

# ============================================================
# mAP LINE PLOT
# ============================================================

plt.figure(figsize=(8,5))

map_values = [

    results[m]["mAP"]
    for m in sorted_methods

]

plt.plot(
    sorted_methods,
    map_values,
    marker='o'
)

for i, v in enumerate(map_values):

    plt.text(
        i,
        v,
        f"{v:.3f}"
    )

plt.ylabel("mAP")
plt.xlabel("Retrieval Method")
plt.title("mAP Comparison")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "mAP_lineplot.png"
    ),

    dpi=300

)

plt.close()

# ============================================================
# LATENCY LINE PLOT
# ============================================================

plt.figure(figsize=(8,5))

lat_values = [

    results[m]["latency"]
    for m in sorted_methods

]

plt.plot(
    sorted_methods,
    lat_values,
    marker='o'
)

for i, v in enumerate(lat_values):

    plt.text(
        i,
        v,
        f"{v:.2f}"
    )

plt.ylabel("Latency (ms)")
plt.xlabel("Retrieval Method")

plt.title(
    "Latency Comparison"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "latency_lineplot.png"
    ),

    dpi=300

)

plt.close()

# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "metrics_summary.txt"
)

with open(summary_path, "w") as f:

    for method, vals in results.items():

        f.write(f"\n{method}\n")

        for k, v in vals.items():

            f.write(f"{k}: {v:.4f}\n")

# ============================================================
# FINAL PRINT
# ============================================================

print("\n================================================")
print("FINAL RESULTS")
print("================================================")

for method, vals in results.items():

    print(f"\n{method}")

    for k, v in vals.items():

        print(f"{k}: {v:.4f}")

print("\nFinished.")
print("Results saved to:")
print(RESULTS_DIR)