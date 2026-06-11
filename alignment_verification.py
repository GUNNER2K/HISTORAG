import os
import json
import random
import h5py
import numpy as np
import openslide
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.geometry import shape
from shapely.ops import unary_union

# ============================================================
# PATHS
# ============================================================

H5_PATH = "/home/woody/iwi5/iwi5411h/BIMAP/data/patches_PT_484/uni_dense/20x_256px_128px_overlap/features_uni_v1/PrimaryTumor_HE_484.h5"

WSI_PATH = "/home/woody/iwi5/iwi5411h/BIMAP/data/WSI_PrimaryTumor_CUP/PrimaryTumor_HE_484.svs"

GEOJSON_PATH = "/home/woody/iwi5/iwi5411h/BIMAP/data/WSI_PrimaryTumor_Annotations/PrimaryTumor_HE_484.geojson"

RESULTS_DIR = "/home/woody/iwi5/iwi5411h/BIMAP/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

PATCH_SIZE = 256

# ============================================================
# LOAD H5
# ============================================================

print("Loading H5...")

with h5py.File(H5_PATH, "r") as f:

    coords = f["coords"][:]
    features = f["features"][:]

print("Coords:", coords.shape)
print("Features:", features.shape)

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

print("Tumor polygons loaded")

# ============================================================
# LOAD WSI
# ============================================================

print("Loading WSI...")

slide = openslide.OpenSlide(WSI_PATH)

w, h = slide.dimensions

print("Slide dimensions:", w, h)

# ============================================================
# LABEL PATCHES
# ============================================================

print("Assigning labels...")

labels = []

for x, y in coords:

    center_x = x 
    center_y = y

    pt = Point(center_x, center_y)

    inside = tumor_region.contains(pt)

    labels.append(int(inside))

labels = np.array(labels)

num_tumor = np.sum(labels == 1)
num_bg = np.sum(labels == 0)

print("Tumor patches:", num_tumor)
print("Background patches:", num_bg)

# ============================================================
# SAVE STATS
# ============================================================

stats_path = os.path.join(
    RESULTS_DIR,
    "alignment_stats.txt"
)

with open(stats_path, "w") as f:

    f.write(f"Total patches: {len(labels)}\n")
    f.write(f"Tumor patches: {num_tumor}\n")
    f.write(f"Background patches: {num_bg}\n")
    f.write(
        f"Tumor percentage: {100*num_tumor/len(labels):.2f}%\n"
    )

print("Stats saved")

# ============================================================
# THUMBNAIL
# ============================================================

thumb_size = (3000, 3000)

thumb = slide.get_thumbnail(
    thumb_size
)

thumb = np.array(thumb)

scale_x = thumb.shape[1] / w
scale_y = thumb.shape[0] / h

# ============================================================
# OVERVIEW FIGURE
# ============================================================

print("Generating overview...")

plt.figure(figsize=(12,12))

plt.imshow(thumb)

# polygons

for poly in polygons:

    if poly.geom_type == "Polygon":

        coords_poly = np.array(
            poly.exterior.coords
        )

        plt.plot(
            coords_poly[:,0] * scale_x,
            coords_poly[:,1] * scale_y,
            color="yellow",
            linewidth=1
        )

    elif poly.geom_type == "MultiPolygon":

        for p in poly.geoms:

            coords_poly = np.array(
                p.exterior.coords
            )

            plt.plot(
                coords_poly[:,0] * scale_x,
                coords_poly[:,1] * scale_y,
                color="yellow",
                linewidth=1
            )

# patch centers

sample_idx = np.random.choice(
    len(coords),
    min(500, len(coords)),
    replace=False
)

sample_coords = coords[sample_idx]

plt.scatter(
    sample_coords[:,0] * scale_x,
    sample_coords[:,1] * scale_y,
    s=5,
    c="cyan"
)

plt.title(
    "WSI + Tumor Annotation + Patch Centers"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "alignment_overview.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# LABEL VISUALIZATION
# ============================================================

print("Generating label overlay...")

plt.figure(figsize=(12,12))

plt.imshow(thumb)

tumor_pts = coords[labels == 1]
bg_pts = coords[labels == 0]

plt.scatter(
    bg_pts[:,0] * scale_x,
    bg_pts[:,1] * scale_y,
    s=6,
    c="red",
    label="Background"
)

plt.scatter(
    tumor_pts[:,0] * scale_x,
    tumor_pts[:,1] * scale_y,
    s=6,
    c="lime",
    label="Tumor"
)

plt.legend()

plt.title(
    "Patch Labels from GeoJSON"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "alignment_labels.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# PATCH EXAMPLES
# ============================================================

print("Generating patch examples...")

tumor_indices = np.where(labels == 1)[0]
bg_indices = np.where(labels == 0)[0]

tumor_examples = np.random.choice(
    tumor_indices,
    min(5, len(tumor_indices)),
    replace=False
)

bg_examples = np.random.choice(
    bg_indices,
    min(5, len(bg_indices)),
    replace=False
)

fig, axes = plt.subplots(
    2,
    5,
    figsize=(15,6)
)

for i, idx in enumerate(tumor_examples):

    x, y = coords[idx]

    patch = slide.read_region(
        (int(x), int(y)),
        0,
        (PATCH_SIZE, PATCH_SIZE)
    ).convert("RGB")

    axes[0, i].imshow(patch)
    axes[0, i].set_title("Tumor")
    axes[0, i].axis("off")

for i, idx in enumerate(bg_examples):

    x, y = coords[idx]

    patch = slide.read_region(
        (int(x), int(y)),
        0,
        (PATCH_SIZE, PATCH_SIZE)
    ).convert("RGB")

    axes[1, i].imshow(patch)
    axes[1, i].set_title("Background")
    axes[1, i].axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "patch_examples.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(6,4))

plt.bar(
    ["Tumor","Background"],
    [num_tumor, num_bg]
)

plt.ylabel("Number of patches")

plt.title("Patch Label Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "class_distribution.png"
    ),
    dpi=300
)

plt.close()

print("\nFinished.")
print("Results saved to:")
print(RESULTS_DIR)