import os
import h5py
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DEMO_DIR = ROOT / "demo_data"

DEMO_DIR.mkdir(
    exist_ok=True,
    parents=True
)

# ============================================================
# ORIGINAL H5 FILES
# ============================================================

FEATURE_PATHS = {

    "sample_uni.h5":

    "/home/woody/iwi5/iwi5411h/BIMAP/data/"
    "patches_PT_484/uni_dense/"
    "20x_256px_128px_overlap/"
    "features_uni_v1/"
    "PrimaryTumor_HE_484.h5",

    "sample_uni2.h5":

    "/home/woody/iwi5/iwi5411h/BIMAP/data/"
    "patches_PT_484/uni2/"
    "20x_256px_128px_overlap/"
    "features_uni_v2/"
    "PrimaryTumor_HE_484.h5",

    "sample_conch.h5":

    "/home/woody/iwi5/iwi5411h/BIMAP/data/"
    "patches_PT_484/conch/"
    "20x_256px_128px_overlap/"
    "features_conch_v15/"
    "PrimaryTumor_HE_484.h5",

    "sample_virchow.h5":

    "/home/woody/iwi5/iwi5411h/BIMAP/data/"
    "patches_PT_484/virchow/"
    "20x_256px_128px_overlap/"
    "features_virchow/"
    "PrimaryTumor_HE_484.h5"

}

# ============================================================
# SAME CROP USED FOR sample_wsi.png
# ============================================================

CROP_X = 10000
CROP_Y = 65000

CROP_W = 8192
CROP_H = 8192

# ============================================================
# PROCESS EACH MODEL
# ============================================================

for output_name, h5_path in FEATURE_PATHS.items():

    print("\n================================================")
    print("Processing:", output_name)
    print("================================================")

    # --------------------------------------------------------
    # LOAD ORIGINAL H5
    # --------------------------------------------------------

    with h5py.File(h5_path, "r") as f:

        coords = f["coords"][:]

        features = f["features"][:]

    print("Original coords:", coords.shape)
    print("Original features:", features.shape)

    # --------------------------------------------------------
    # FILTER INSIDE CROP
    # --------------------------------------------------------

    mask = (

        (coords[:,0] >= CROP_X) &
        (coords[:,0] <  CROP_X + CROP_W) &

        (coords[:,1] >= CROP_Y) &
        (coords[:,1] <  CROP_Y + CROP_H)

    )

    coords_crop = coords[mask]
    features_crop = features[mask]

    print("Crop coords:", coords_crop.shape)

    # --------------------------------------------------------
    # SHIFT COORDINATES
    # --------------------------------------------------------

    coords_crop = coords_crop.copy()

    coords_crop[:,0] -= CROP_X
    coords_crop[:,1] -= CROP_Y

    # --------------------------------------------------------
    # RANDOM SAMPLE
    # --------------------------------------------------------

    if len(coords_crop) > 1000:

        idx = np.random.choice(

            len(coords_crop),
            1000,
            replace=False

        )

        coords_crop = coords_crop[idx]
        features_crop = features_crop[idx]

    print("Final sample:", coords_crop.shape)

    # --------------------------------------------------------
    # SAVE DEMO H5
    # --------------------------------------------------------

    out_path = DEMO_DIR / output_name

    with h5py.File(out_path, "w") as f:

        f.create_dataset(
            "coords",
            data=coords_crop
        )

        f.create_dataset(
            "features",
            data=features_crop
        )

    print("Saved:", out_path)

print("\nDONE")