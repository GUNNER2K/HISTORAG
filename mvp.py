import os
import time
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
import openslide

from PIL import Image
from pathlib import Path

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

RESULTS_DIR = Path(RESULTS_DIR) / "mvp"

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_h5(path):

    print(f"\nLoading features from:\n{path}")

    with h5py.File(path, 'r') as f:
        features = f['features'][:]
        coords = f['coords'][:]

    print("Features shape:", features.shape)
    print("Coords shape:", coords.shape)

    return features, coords

all_features = []
all_coords = []
all_slide_ids = []

slide_objects = {}
slide_paths = {}

if not USE_DEMO == False:

    print("\n================================================")
    print("RUNNING DEMO MODE")
    print("================================================")

    X, coords = load_h5(FEATURE_PATHS['UNI'])

    all_features.append(X)

    all_coords.append(coords)

    all_slide_ids.append(np.zeros(len(X)))

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

            stem = h5_file.stem

            svs_path = WSI_ROOT / f"{stem}.svs"

            if not svs_path.exists():

                print(f"Missing WSI:\n{svs_path}")

                continue

            features, coords = load_h5(h5_file)

            all_features.append(features)

            all_coords.append(coords)

            all_slide_ids.append(np.full(len(features), slide_idx))

            slide = openslide.OpenSlide(str(svs_path))

            slide_objects[slide_idx] = slide

            slide_paths[slide_idx] = svs_path

        except Exception as e:

            print(f"\nSkipping {h5_file.name}")

            print(e)

        if slide_idx % 10 == 0:
            print('\nProcessed 10 files...')

X = np.vstack(all_features)

coords = np.vstack(all_coords)

slide_ids = np.concatenate(all_slide_ids)

print("\n================================================")
print("FINAL DATASET")
print("================================================")

print("Total patches:", len(X))
print("Feature dimension:", X.shape[1])

print("\nNormalizing features")

X = X.astype(np.float32)

X = X / np.linalg.norm(X, axis=1, keepdims=True)

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

def retrieve(query_idx, top_k=10):

    print(f"\nRunning retrieval for query index: {query_idx}")

    q = X[query_idx]

    sims = X @ q

    indices = np.argsort(-sims)[:top_k + 1]

    indices = indices[indices != query_idx][:top_k]

    print("Retrieved indices:", indices)

    return indices

def visualize(query_idx, retrieved_indices, save_path):

    print(f"\nSaving visualization:\n{save_path}")

    fig, axes = plt.subplots(1, TOP_K + 1, figsize=(20, 4))

    axes[0].imshow(get_patch(query_idx))

    axes[0].set_title("QUERY")

    axes[0].axis("off")

    for i, idx in enumerate(retrieved_indices):

        axes[i + 1].imshow(get_patch(idx))

        slide_name = Path(str(slide_paths[slide_ids[idx]])).stem

        axes[i + 1].set_title(f"Top-{i+1}")

        axes[i + 1].axis("off")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.close()

start_time = time.time()

print("\nStarting MVP retrieval...\n")

for i in range(NUM_QUERIES):

    print(f"\n================ QUERY {i+1} ================")

    query_idx = np.random.randint(0, len(X))

    retrieved_indices = retrieve(query_idx, top_k=TOP_K)

    save_path = RESULTS_DIR / f"query_{i+1}.png"

    visualize(query_idx, retrieved_indices, save_path)

end_time = time.time()

print("\n================================================")
print("FINISHED")
print("================================================")

print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

print(f"\nResults saved in:\n{RESULTS_DIR}")