import os
import time
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
import openslide

from PIL import Image
from pathlib import Path

# ============================================================
# CONFIG SELECTION
# ============================================================

USE_DEMO = False

if USE_DEMO:

    from configs.demo_config import *

else:

    from configs.full_config import *

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = Path(RESULTS_DIR) / "mvp"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# ============================================================
# LOAD H5
# ============================================================

def load_h5(path):

    print(f"\nLoading features from:\n{path}")

    with h5py.File(path, 'r') as f:

        features = f['features'][:]
        coords = f['coords'][:]

    print("Features shape:", features.shape)
    print("Coords shape:", coords.shape)

    return features, coords

# ============================================================
# STORAGE
# ============================================================

all_features = []
all_coords = []
all_slide_ids = []

slide_objects = {}
slide_paths = {}

# ============================================================
# DEMO MODE
# ============================================================

if not USE_DEMO == False:

    print("\n================================================")
    print("RUNNING DEMO MODE")
    print("================================================")

    X, coords = load_h5(H5_PATH)

    all_features.append(X)

    all_coords.append(coords)

    all_slide_ids.append(
        np.zeros(len(X))
    )

    # --------------------------------------------------------
    # LOAD DEMO IMAGE
    # --------------------------------------------------------

    demo_img = Image.open(
        WSI_PATH
    ).convert("RGB")

    demo_img = np.array(
        demo_img
    )

    slide_objects[0] = demo_img

    slide_paths[0] = WSI_PATH

# ============================================================
# FULL MODE
# ============================================================

else:

    print("\n================================================")
    print("RUNNING FULL MODE")
    print("================================================")

    # --------------------------------------------------------
    # FIND ALL H5 FILES
    # --------------------------------------------------------

    all_h5_files = list(
        H5_ROOT.rglob("*.h5")
    )

    print(f"\nFound {len(all_h5_files)} H5 files")

    if MAX_WSIS is not None:

        all_h5_files = all_h5_files[:MAX_WSIS]

    # --------------------------------------------------------
    # LOAD EACH WSI
    # --------------------------------------------------------

    for slide_idx, h5_file in enumerate(all_h5_files):

        try:

            # ------------------------------------------------
            # MATCH SVS
            # ------------------------------------------------

            stem = h5_file.stem

            svs_path = (

                WSI_ROOT
                / f"{stem}.svs"

            )

            if not svs_path.exists():

                print(f"Missing WSI:\n{svs_path}")

                continue

            # ------------------------------------------------
            # LOAD FEATURES
            # ------------------------------------------------

            features, coords = load_h5(h5_file)

            all_features.append(
                features
            )

            all_coords.append(
                coords
            )

            all_slide_ids.append(

                np.full(
                    len(features),
                    slide_idx
                )

            )

            # ------------------------------------------------
            # LOAD WSI
            # ------------------------------------------------

            slide = openslide.OpenSlide(
                str(svs_path)
            )

            slide_objects[slide_idx] = slide

            slide_paths[slide_idx] = svs_path


        except Exception as e:

            print(f"\nSkipping {h5_file.name}")

            print(e)

        if slide_idx % 10 == 0:
            print('\nProcessed 10 files...')
# ============================================================
# CONCATENATE
# ============================================================

X = np.vstack(
    all_features
)

coords = np.vstack(
    all_coords
)

slide_ids = np.concatenate(
    all_slide_ids
)

print("\n================================================")
print("FINAL DATASET")
print("================================================")

print("Total patches:", len(X))
print("Feature dimension:", X.shape[1])

# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing features")

X = X.astype(np.float32)

X = X / np.linalg.norm(

    X,
    axis=1,
    keepdims=True

)

# ============================================================
# PATCH EXTRACTION
# ============================================================

def get_patch(idx):

    x, y = coords[idx]

    x = int(x)
    y = int(y)

    slide_id = slide_ids[idx]

    # --------------------------------------------------------
    # DEMO MODE
    # --------------------------------------------------------

    if USE_DEMO:

        img = slide_objects[slide_id]

        patch = img[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        return patch

    # --------------------------------------------------------
    # FULL MODE
    # --------------------------------------------------------

    else:

        slide = slide_objects[slide_id]

        patch = slide.read_region(

            (x, y),
            0,
            (PATCH_SIZE, PATCH_SIZE)

        ).convert("RGB")

        return np.array(patch)

# ============================================================
# BRUTE FORCE RETRIEVAL
# ============================================================

def retrieve(query_idx, top_k=10):

    print(f"\nRunning retrieval for query index: {query_idx}")

    q = X[query_idx]

    sims = X @ q

    indices = np.argsort(
        -sims
    )[:top_k + 1]

    # --------------------------------------------------------
    # REMOVE QUERY ITSELF
    # --------------------------------------------------------

    indices = indices[
        indices != query_idx
    ][:top_k]

    print("Retrieved indices:", indices)

    return indices

# ============================================================
# VISUALIZATION
# ============================================================

def visualize(

    query_idx,
    retrieved_indices,
    save_path

):

    print(f"\nSaving visualization:\n{save_path}")

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
    # RETRIEVED
    # --------------------------------------------------------

    for i, idx in enumerate(retrieved_indices):

        axes[i + 1].imshow(
            get_patch(idx)
        )

        slide_name = Path(

            str(
                slide_paths[
                    slide_ids[idx]
                ]
            )

        ).stem

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
# RUN MVP
# ============================================================

start_time = time.time()

print("\nStarting MVP retrieval...\n")

for i in range(NUM_QUERIES):

    print(f"\n================ QUERY {i+1} ================")

    query_idx = np.random.randint(
        0,
        len(X)
    )

    retrieved_indices = retrieve(

        query_idx,
        top_k=TOP_K

    )

    save_path = (

        RESULTS_DIR
        / f"query_{i+1}.png"

    )

    visualize(

        query_idx,
        retrieved_indices,
        save_path

    )

# ============================================================
# FINAL
# ============================================================

end_time = time.time()

print("\n================================================")
print("FINISHED")
print("================================================")

print(
    f"\nTotal runtime: "
    f"{end_time - start_time:.2f} seconds"
)

print(f"\nResults saved in:\n{RESULTS_DIR}")