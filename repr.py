import os
import json
import h5py
import numpy as np

from shapely.geometry import Point
from shapely.geometry import shape
from shapely.ops import unary_union

# ============================================================
# INPUT PATHS
# ============================================================

MODEL_PATHS = {

    "conch":
    "/home/woody/iwi5/iwi5411h/BIMAP/data/patches_PT_484/conch/20x_256px_128px_overlap/features_conch_v15/PrimaryTumor_HE_484.h5",

    "uni2":
    "/home/woody/iwi5/iwi5411h/BIMAP/data/patches_PT_484/uni2/20x_256px_128px_overlap/features_uni_v2/PrimaryTumor_HE_484.h5",

    "virchow":
    "/home/woody/iwi5/iwi5411h/BIMAP/data/patches_PT_484/virchow/20x_256px_128px_overlap/features_virchow/PrimaryTumor_HE_484.h5",

    'uni':
    "/home/woody/iwi5/iwi5411h/BIMAP/data/patches_PT_484/uni_dense/20x_256px_128px_overlap/features_uni_v1/PrimaryTumor_HE_484.h5"
}

GEOJSON_PATH = (
    "/home/woody/iwi5/iwi5411h/BIMAP/data/"
    "WSI_PrimaryTumor_Annotations/"
    "PrimaryTumor_HE_484.geojson"
)

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = "/home/hpc/iwi5/iwi5411h/bimap/demo_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

PATCH_SIZE = 256

TOTAL_SAMPLES = 1000

SEED = 42

np.random.seed(SEED)

# ============================================================
# LOAD REFERENCE COORDS
# ============================================================

print("Loading reference coordinates...")

with h5py.File(MODEL_PATHS["uni2"], "r") as f:

    coords = f["coords"][:]

print("Total patches:", len(coords))

# ============================================================
# LOAD GEOJSON
# ============================================================

print("Loading GeoJSON...")

with open(GEOJSON_PATH, "r") as f:

    geo = json.load(f)

polygons = []

for feature in geo["features"]:

    geom = shape(feature["geometry"])

    polygons.append(geom)

tumor_region = unary_union(polygons)

# ============================================================
# LABEL PATCHES
# ============================================================

print("Assigning labels...")

labels = []

for x, y in coords:

    center_x = x + PATCH_SIZE / 2
    center_y = y + PATCH_SIZE / 2

    pt = Point(center_x, center_y)

    inside = tumor_region.contains(pt)

    labels.append(int(inside))

labels = np.array(labels)

tumor_indices = np.where(labels == 1)[0]
bg_indices = np.where(labels == 0)[0]

print("Tumor patches:", len(tumor_indices))
print("Background patches:", len(bg_indices))

# ============================================================
# SAMPLE BACKGROUND
# ============================================================

num_bg_needed = TOTAL_SAMPLES - len(tumor_indices)

sampled_bg = np.random.choice(
    bg_indices,
    num_bg_needed,
    replace=False
)

# ============================================================
# FINAL INDICES
# ============================================================

final_indices = np.concatenate([
    tumor_indices,
    sampled_bg
])

final_indices = np.sort(final_indices)

print("Final sampled patches:", len(final_indices))

# ============================================================
# SAVE SAMPLED FILES
# ============================================================

for model_name, path in MODEL_PATHS.items():

    print(f"\nProcessing {model_name}")

    with h5py.File(path, "r") as f:

        coords = f["coords"][:]
        features = f["features"][:]

    sampled_coords = coords[final_indices]
    sampled_features = features[final_indices]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{model_name}.h5"
    )

    with h5py.File(output_path, "w") as f:

        f.create_dataset(
            "coords",
            data=sampled_coords
        )

        f.create_dataset(
            "features",
            data=sampled_features
        )

        f.create_dataset(
            "labels",
            data=labels[final_indices]
        )

    print("Saved:", output_path)

    print("Coords:", sampled_coords.shape)
    print("Features:", sampled_features.shape)

# ============================================================
# COPY GEOJSON
# ============================================================

import shutil

shutil.copy(
    GEOJSON_PATH,
    os.path.join(
        OUTPUT_DIR,
        "sample_annotations.geojson"
    )
)

print("\nDone.")
print("Demo dataset saved to:")
print(OUTPUT_DIR)