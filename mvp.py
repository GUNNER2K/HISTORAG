import os
import time
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

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

RESULTS_DIR = (
    ROOT_DIR
    / "results"
    / "mvp"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

PATCH_SIZE = 256
TOP_K = 10
NUM_QUERIES = 5

# ============================================================
# LOAD H5
# ============================================================

def load_h5(path):

    print(f"\nLoading features from: {path}")

    with h5py.File(path, 'r') as f:

        features = f['features'][:]
        coords = f['coords'][:]

    print(
        "Features shape:",
        features.shape
    )

    print(
        "Coords shape:",
        coords.shape
    )

    return features, coords

# ============================================================
# START TIMER
# ============================================================

start_time = time.time()

# ============================================================
# LOAD FEATURES
# ============================================================

X, coords = load_h5(H5_PATH)

# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing features...")

X = X.astype(np.float32)

X = X / np.linalg.norm(
    X,
    axis=1,
    keepdims=True
)

# ============================================================
# LOAD DEMO WSI PNG
# ============================================================

print("\nLoading sample_wsi.png...")

wsi_img = Image.open(
    WSI_PATH
).convert("RGB")

wsi_img = np.array(wsi_img)

print(
    "Demo image shape:",
    wsi_img.shape
)

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
# BRUTE FORCE RETRIEVAL
# ============================================================

def retrieve(query_idx, top_k=10):

    print(
        f"\nRunning retrieval "
        f"for query index: {query_idx}"
    )

    q = X[query_idx]

    sims = X @ q

    indices = np.argsort(
        -sims
    )[:top_k + 1]

    # remove query itself

    indices = indices[
        indices != query_idx
    ][:top_k]

    print(
        "Retrieved indices:",
        indices
    )

    return indices

# ============================================================
# VISUALIZATION
# ============================================================

def visualize(

    query_idx,
    retrieved_indices,
    save_path

):

    print(
        f"\nSaving visualization to:"
        f"\n{save_path}"
    )

    fig, axes = plt.subplots(

        1,
        TOP_K + 1,

        figsize=(20, 4)

    )

    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    axes[0].imshow(
        get_patch(query_idx)
    )

    axes[0].set_title("QUERY")

    axes[0].axis("off")

    # --------------------------------------------------------
    # RETRIEVED PATCHES
    # --------------------------------------------------------

    for i, idx in enumerate(retrieved_indices):

        axes[i + 1].imshow(
            get_patch(idx)
        )

        axes[i + 1].set_title(
            f"Top-{i+1}"
        )

        axes[i + 1].axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()

# ============================================================
# RETRIEVAL EXPERIMENTS
# ============================================================

print("\nStarting MVP retrieval...\n")

for i in range(NUM_QUERIES):

    print(
        f"\n================ "
        f"QUERY {i+1} "
        f"================"
    )

    query_idx = np.random.randint(
        0,
        len(X)
    )

    retrieved_indices = retrieve(
        query_idx,
        top_k=TOP_K
    )

    save_path = os.path.join(

        RESULTS_DIR,
        f"query_{i+1}.png"

    )

    visualize(

        query_idx,
        retrieved_indices,
        save_path

    )

# ============================================================
# FINISH
# ============================================================

end_time = time.time()

print("\nFinished all queries.")

print(
    f"Total runtime: "
    f"{end_time - start_time:.2f} seconds"
)

print(
    f"Results saved in:\n"
    f"{RESULTS_DIR}"
)