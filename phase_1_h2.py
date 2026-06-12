import os
import json
import time
import h5py
import faiss
import random
import numpy as np
import matplotlib.pyplot as plt
import openslide

from PIL import Image
from pathlib import Path

from shapely.geometry import Point
from shapely.geometry import shape
from shapely.ops import unary_union

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--demo", action="store_true", help="Run in demo mode")

args = parser.parse_args()

USE_DEMO = args.demo

if USE_DEMO:
    from configs.demo_config import *
else:
    from configs.full_config import *

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

RESULTS_DIR = Path(RESULTS_DIR) / "h2"

os.makedirs(RESULTS_DIR, exist_ok=True)

all_features = []
all_coords = []
all_labels = []
all_slide_ids = []

slide_objects = {}
slide_paths = {}

def load_h5(path):

    with h5py.File(path, "r") as f:
        coords = f["coords"][:]
        features = f["features"][:]

    return coords, features

def load_geojson(path):

    with open(path, "r") as f:
        geo = json.load(f)

    polygons = []

    for feature in geo["features"]:

        geom = shape(feature["geometry"])

        polygons.append(geom)

    return unary_union(polygons)

def assign_labels(coords, tumor_region):

    labels = []

    for x, y in coords:

        center_x = x + PATCH_SIZE / 2
        center_y = y + PATCH_SIZE / 2

        pt = Point(center_x, center_y)

        inside = tumor_region.contains(pt)

        labels.append(int(inside))

    return np.array(labels)

if USE_DEMO:

    print("\n================================================")
    print("RUNNING DEMO MODE")
    print("================================================")

    coords, features = load_h5(FEATURE_PATHS['UNI'])

    tumor_region = load_geojson(GEOJSON_PATH)

    labels = assign_labels(coords, tumor_region)

    all_features.append(features)

    all_coords.append(coords)

    all_labels.append(labels)

    all_slide_ids.append(np.zeros(len(features)))

    demo_img = Image.open(WSI_SMALL_PATH).convert("RGB")

    demo_img = np.array(demo_img)

    slide_objects[0] = demo_img

    slide_paths[0] = WSI_SMALL_PATH

else:

    print("\n================================================")
    print("RUNNING FULL MODE")
    print("================================================")

    all_h5_files = list(H5_ROOT.rglob("*.h5"))

    print(f"\nFound {len(all_h5_files)} H5 files")

    if MAX_WSIS is not None:
        all_h5_files = all_h5_files[:MAX_WSIS]

    for slide_idx, h5_file in enumerate(all_h5_files):

        try:

            print("\n================================================")
            print(f"PROCESSING:\n{h5_file.name}")
            print("================================================")

            stem = h5_file.stem

            geojson_path = ANNOTATION_ROOT / f"{stem}.geojson"

            if not geojson_path.exists():

                print("Missing annotation")

                continue

            svs_path = WSI_ROOT / f"{stem}.svs"

            if not svs_path.exists():

                print("Missing WSI")

                continue

            coords, features = load_h5(h5_file)

            tumor_region = load_geojson(geojson_path)

            labels = assign_labels(coords, tumor_region)

            tumor_count = np.sum(labels == 1)

            if tumor_count < MIN_TUMOR_PATCHES:

                print(f"Skipping {stem} (tumor patches={tumor_count})")

                continue

            print(f"Tumor patches: {tumor_count}")

            all_features.append(features)

            all_coords.append(coords)

            all_labels.append(labels)

            all_slide_ids.append(np.full(len(features), slide_idx))

            slide = openslide.OpenSlide(str(svs_path))

            slide_objects[slide_idx] = slide

            slide_paths[slide_idx] = svs_path

            print("WSI loaded")

        except Exception as e:

            print(f"\nSkipping {h5_file.name}")

            print(e)

features = np.vstack(all_features)

coords = np.vstack(all_coords)

labels = np.concatenate(all_labels)

slide_ids = np.concatenate(all_slide_ids)

print("\n================================================")
print("FINAL DATASET")
print("================================================")

print("Total patches:", len(labels))

print("Tumor patches:", np.sum(labels == 1))

print("Background patches:", np.sum(labels == 0))

features = features.astype(np.float32)

features = features / np.linalg.norm(features, axis=1, keepdims=True)

tumor_indices = np.where(labels == 1)[0]

query_indices = np.random.choice(tumor_indices, min(NUM_QUERIES, len(tumor_indices)), replace=False)

print(f"\nUsing {len(query_indices)} tumor queries")

def get_patch(idx):

    x, y = coords[idx]

    x = int(x)
    y = int(y)

    slide_id = slide_ids[idx]

    if USE_DEMO:

        img = slide_objects[slide_id]

        patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

        return patch

    else:

        slide = slide_objects[slide_id]

        patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")

        return np.array(patch)

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

            precisions.append(relevant_count / rank)

    if len(precisions) == 0:
        return 0

    return np.mean(precisions)

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

        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

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

        self.index = faiss.IndexHNSWFlat(d, 32)

        self.index.hnsw.efConstruction = 40

        self.index.add(X)

    def search(self, q, k):

        q = q.reshape(1, -1)

        scores, idx = self.index.search(q, k)

        return idx[0], scores[0]

methods = {
    "BruteForce": BruteForce(features),
    "FAISSFlat": FAISSFlat(features),
    "FAISSIVF": FAISSIVF(features),
    "FAISSHNSW": FAISSHNSW(features)
}

results = {}

for name, method in methods.items():

    print(f"\nRunning {name}")

    p5_scores = []
    p10_scores = []
    map_scores = []
    latency_scores = []

    for query_idx in query_indices:

        query_embedding = features[query_idx]

        start = time.time()

        retrieved_indices, scores = method.search(query_embedding, TOP_K + 1)

        latency = (time.time() - start) * 1000

        retrieved_indices = retrieved_indices[retrieved_indices != query_idx][:TOP_K]

        p5 = precision_at_k(retrieved_indices, labels, 5)

        p10 = precision_at_k(retrieved_indices, labels, 10)

        ap = average_precision(retrieved_indices, labels)

        p5_scores.append(p5)
        p10_scores.append(p10)
        map_scores.append(ap)
        latency_scores.append(latency)

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

    fixed_query = query_indices[0]

    query_embedding = features[fixed_query]

    retrieved_indices, scores = method.search(query_embedding, TOP_K + 1)

    retrieved_indices = retrieved_indices[retrieved_indices != fixed_query][:TOP_K]

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    axes = axes.flatten()

    axes[0].imshow(get_patch(fixed_query))

    axes[0].set_title("QUERY")

    axes[0].axis("off")

    for i, idx in enumerate(retrieved_indices):

        axes[i + 1].imshow(get_patch(idx))

        label = "Tumor" if labels[idx] == 1 else "BG"

        axes[i + 1].set_title(label)

        axes[i + 1].axis("off")

    for j in range(len(retrieved_indices) + 1, 16):
        axes[j].axis("off")

    plt.suptitle(f"{name}\nAVG P@5={avg_p5:.2f} | AVG P@10={avg_p10:.2f} | AVG mAP={avg_map:.2f} | AVG Latency={avg_latency:.2f}ms")

    plt.tight_layout()

    plt.savefig(RESULTS_DIR / f"{name}_retrieval.png", dpi=300)

    plt.close()

sorted_methods = sorted(results.keys(), key=lambda x: results[x]["latency"], reverse=True)

plt.figure(figsize=(8, 5))

map_values = [results[m]["mAP"] for m in sorted_methods]

plt.plot(sorted_methods, map_values, marker='o')

for i, v in enumerate(map_values):
    plt.text(i, v, f"{v:.3f}")

plt.ylabel("mAP")

plt.xlabel("Retrieval Method")

plt.title("mAP Comparison")

plt.grid(True)

plt.tight_layout()

plt.savefig(RESULTS_DIR / "mAP_lineplot.png", dpi=300)

plt.close()

plt.figure(figsize=(8, 5))

lat_values = [results[m]["latency"] for m in sorted_methods]

plt.plot(sorted_methods, lat_values, marker='o')

for i, v in enumerate(lat_values):
    plt.text(i, v, f"{v:.2f}")

plt.ylabel("Latency (ms)")

plt.xlabel("Retrieval Method")

plt.title("Latency Comparison")

plt.grid(True)

plt.tight_layout()

plt.savefig(RESULTS_DIR / "latency_lineplot.png", dpi=300)

plt.close()

summary_path = RESULTS_DIR / "metrics_summary.txt"

with open(summary_path, "w") as f:

    for method, vals in results.items():

        f.write(f"\n{method}\n")

        for k, v in vals.items():
            f.write(f"{k}: {v:.4f}\n")

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