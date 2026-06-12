from pathlib import Path

# ============================================================
# ROOT DIRECTORIES
# ============================================================

H5_ROOT = Path(

    "/home/woody/iwi5/iwi5411h/BIMAP/data/patches_PT_484/pre_uni/WSI_PrimaryTumor/WSI_PrimaryTumor_Hypopharynx/h5_files/"

)

ANNOTATION_ROOT = Path(

    "/home/woody/iwi5/iwi5411h/BIMAP/data/"
    "WSI_PrimaryTumor_Annotations/"

)

# ============================================================
# RESULTS
# ============================================================

RESULTS_DIR = Path(

    "/home/woody/iwi5/iwi5411h/BIMAP/results"

)

# ============================================================
# PARAMETERS
# ============================================================

PATCH_SIZE = 256

TOP_K = 15

NUM_QUERIES = 50

RANDOM_SEED = 42

# ============================================================
# DATASET SETTINGS
# ============================================================

MAX_WSIS = None

MIN_TUMOR_PATCHES = 5

# ============================================================
# VISUALIZATION
# ============================================================

USE_TSNE = True

USE_3D = True

SAVE_VISUALIZATIONS = True