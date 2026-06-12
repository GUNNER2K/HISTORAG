import os
import json
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots

from shapely.geometry import Point
from shapely.geometry import shape
from shapely.ops import unary_union

random.seed(42)
np.random.seed(42)

ROOT_DIR = Path(__file__).resolve().parent

FEATURE_PATHS = {
    "UNI": ROOT_DIR / "demo_data" / "sample_uni.h5",
    "UNI2": ROOT_DIR / "demo_data" / "sample_uni2.h5",
    "CONCH": ROOT_DIR / "demo_data" / "sample_conch.h5",
    "VIRCHOW": ROOT_DIR / "demo_data" / "sample_virchow.h5"
}

GEOJSON_PATH = ROOT_DIR / "demo_data" / "sample_annotations.geojson"

RESULTS_DIR = ROOT_DIR / "results" / "h4"

os.makedirs(RESULTS_DIR, exist_ok=True)

PATCH_SIZE = 256

print("\nLoading annotations...")

with open(GEOJSON_PATH, "r") as f:
    geo = json.load(f)

polygons = []

for feature in geo["features"]:

    geom = shape(feature["geometry"])

    polygons.append(geom)

tumor_region = unary_union(polygons)

print("Annotations loaded")

fig_2d = plt.figure(figsize=(16, 12))

fig_3d = make_subplots(
    rows=2,
    cols=2,
    specs=[
        [{"type": "scene"}, {"type": "scene"}],
        [{"type": "scene"}, {"type": "scene"}]
    ],
    subplot_titles=list(FEATURE_PATHS.keys())
)

for idx, (model_name, h5_path) in enumerate(FEATURE_PATHS.items()):

    print("\n================================================")
    print(f"PROCESSING MODEL: {model_name}")
    print("================================================")

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
        features = f["features"][:]

    print("Coords:", coords.shape)
    print("Features:", features.shape)

    features = features.astype(np.float32)

    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    labels = []

    for x, y in coords:

        center_x = x + PATCH_SIZE / 2
        center_y = y + PATCH_SIZE / 2

        pt = Point(center_x, center_y)

        inside = tumor_region.contains(pt)

        labels.append(int(inside))

    labels = np.array(labels)

    tumor_count = np.sum(labels == 1)

    bg_count = np.sum(labels == 0)

    print("Tumor patches:", tumor_count)
    print("Background patches:", bg_count)

    print("Running PCA...")

    pca = PCA(n_components=50, random_state=42)

    features_pca = pca.fit_transform(features)

    print("Running t-SNE 2D...")

    tsne_2d = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='pca',
        random_state=42
    )

    emb_2d = tsne_2d.fit_transform(features_pca)

    print("Running t-SNE 3D...")

    tsne_3d = TSNE(
        n_components=3,
        perplexity=30,
        learning_rate='auto',
        init='pca',
        random_state=42
    )

    emb_3d = tsne_3d.fit_transform(features_pca)

    tumor_mask = labels == 1
    bg_mask = labels == 0

    ax2d = fig_2d.add_subplot(2, 2, idx + 1)

    ax2d.scatter(
        emb_2d[bg_mask, 0],
        emb_2d[bg_mask, 1],
        s=10,
        alpha=0.5,
        label="Background"
    )

    ax2d.scatter(
        emb_2d[tumor_mask, 0],
        emb_2d[tumor_mask, 1],
        s=10,
        alpha=0.9,
        label="Tumor"
    )

    ax2d.set_title(f"{model_name} - 2D t-SNE")

    ax2d.legend()

    row = idx // 2 + 1
    col = idx % 2 + 1

    fig_3d.add_trace(
        go.Scatter3d(
            x=emb_3d[bg_mask, 0],
            y=emb_3d[bg_mask, 1],
            z=emb_3d[bg_mask, 2],
            mode='markers',
            marker=dict(size=3, opacity=0.4),
            name=f"{model_name}-BG",
            showlegend=False
        ),
        row=row,
        col=col
    )

    fig_3d.add_trace(
        go.Scatter3d(
            x=emb_3d[tumor_mask, 0],
            y=emb_3d[tumor_mask, 1],
            z=emb_3d[tumor_mask, 2],
            mode='markers',
            marker=dict(size=4, opacity=0.9),
            name=f"{model_name}-Tumor",
            showlegend=False
        ),
        row=row,
        col=col
    )

fig_2d.tight_layout()

fig_2d.savefig(RESULTS_DIR / "embedding_space_2d.png", dpi=300)

plt.close(fig_2d)

fig_3d.update_layout(
    title="Interactive 3D Embedding Space Comparison",
    height=1200,
    width=1400
)

html_path = RESULTS_DIR / "embedding_space_3d_interactive.html"

fig_3d.write_html(str(html_path))

print("\n================================================")
print("HYPOTHESIS 4 FINISHED")
print("================================================")

print("\nSaved files:")

print(RESULTS_DIR / "embedding_space_2d.png")

print(html_path)