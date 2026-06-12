from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


FEATURE_PATHS = {

    "UNI":
    ROOT_DIR / "demo_data" / "sample_uni.h5",

    "UNI2":
    ROOT_DIR / "demo_data" / "sample_uni2.h5",

    "CONCH":
    ROOT_DIR / "demo_data" / "sample_conch.h5",

    "VIRCHOW":
    ROOT_DIR / "demo_data" / "sample_virchow.h5"

}


GEOJSON_PATH = (

    ROOT_DIR
    / "demo_data"
    / "sample_annotations.geojson"

)


WSI_IMAGE_PATH = (

    ROOT_DIR
    / "demo_data"
    / "small_sample_wsi.jpg"

)

RESULTS_DIR = (

    ROOT_DIR
    / "results"

)


PATCH_SIZE = 256

TOP_K = 15

NUM_QUERIES = 20

RANDOM_SEED = 42

# ============================================================
# VISUALIZATION
# ============================================================

USE_TSNE = True

USE_3D = True

SAVE_VISUALIZATIONS = True