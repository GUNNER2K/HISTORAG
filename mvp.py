import os
import time
import h5py
import numpy as np
import openslide
import matplotlib.pyplot as plt


tumor_feat_path = "/home/woody/iwi5/iwi5411h/BIMAP/patches_tumor/20x_256px_0px_overlap/features_uni_v1/TumorCenter_CD3_block1.h5"
inv_feat_path   = "/home/woody/iwi5/iwi5411h/BIMAP/patches/20x_256px_0px_overlap/features_uni_v1/InvasionFront_CD3_block1.h5"

tumor_wsi_path = "/home/woody/iwi5/iwi5411h/BIMAP/data/TumorCenter_CD3_block1.svs"
inv_wsi_path   = "/home/woody/iwi5/iwi5411h/BIMAP/data/InvasionFront_CD3_block1.svs"

RESULTS_DIR = "/home/woody/iwi5/iwi5411h/BIMAP/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PATCH_SIZE = 256
LEVEL = 0

# -----------------------------
# LOAD H5 FILES
# -----------------------------
def load_h5(path, name):
    print(f"\n[INFO] Loading {name} features from: {path}")
    with h5py.File(path, 'r') as f:
        features = f['features'][:]
        coords = f['coords'][:]
    
    print(f"[INFO] {name} features shape: {features.shape}")
    print(f"[INFO] {name} coords shape:   {coords.shape}")
    
    return features, coords

start_time = time.time()

X_T, C_T = load_h5(tumor_feat_path, "TumorCenter")
X_I, C_I = load_h5(inv_feat_path, "InvasionFront")

# -----------------------------
# CONCATENATE
# -----------------------------
print("\n Concatenating datasets...")

X = np.vstack([X_T, X_I])
coords = np.vstack([C_T, C_I])

slide_ids = np.concatenate([
    np.zeros(len(X_T)),   # 0 = tumor
    np.ones(len(X_I))     # 1 = invasion
])

print(f"Total patches: {len(X)}")
print(f"Feature dimension: {X.shape[1]}")

# -----------------------------
# NORMALIZATION
# -----------------------------
print("\n[INFO] Normalizing feature vectors...")
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# -----------------------------
# LOAD SLIDES
# -----------------------------
print("\n Loading WSI slides...")
slide_T = openslide.OpenSlide(tumor_wsi_path)
slide_I = openslide.OpenSlide(inv_wsi_path)
print(" Slides loaded successfully")

# -----------------------------
# PATCH EXTRACTION
# -----------------------------
def get_patch(idx):
    x, y = coords[idx]
    slide = slide_T if slide_ids[idx] == 0 else slide_I
    
    patch = slide.read_region(
        (int(x), int(y)),
        LEVEL,
        (PATCH_SIZE, PATCH_SIZE)
    ).convert("RGB")
    
    return patch

# -----------------------------
# RETRIEVAL
# -----------------------------
def retrieve(query_idx, top_k=10):
    print(f"\n Running retrieval for query index: {query_idx}")
    
    q = X[query_idx]
    
    sims = X @ q
    indices = np.argsort(-sims)[:top_k]
    
    print(f"Top-{top_k} similarity scores: {sims[indices]}")
    
    return indices, sims[indices]

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate(query_idx, retrieved_indices):
    query_slide = slide_ids[query_idx]
    
    same = sum(slide_ids[i] == query_slide for i in retrieved_indices)
    percent = (same / len(retrieved_indices)) * 100
    
    print("\n Evaluation Results:")
    print(f"Same-slide patches: {same}/{len(retrieved_indices)}")
    print(f"Percentage same region: {percent:.2f}%")
    
    return same, percent

# -----------------------------
# VISUALIZATION (SAVE)
# -----------------------------
def visualize(query_idx, retrieved_indices, save_path):
    print(f"\n[INFO] Saving visualization to: {save_path}")
    
    plt.figure(figsize=(14, 5))
    
    # Query
    plt.subplot(1, len(retrieved_indices)+1, 1)
    plt.imshow(get_patch(query_idx))
    plt.title("Query")
    plt.axis("off")
    
    # Retrieved
    for i, idx in enumerate(retrieved_indices):
        plt.subplot(2, len(retrieved_indices)+1, i+2)
        plt.imshow(get_patch(idx))
        
        label = "Tumor" if slide_ids[idx] == 0 else "Invasion"
        plt.title(label)
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


NUM_QUERIES = 5

print("\n Starting retrieval experiments...\n")

for i in range(NUM_QUERIES):
    print(f"\n================ QUERY {i+1} ================")
    
    query_idx = np.random.randint(0, len(X))
    
    retrieved_indices, scores = retrieve(query_idx, top_k=10)
    
    same_count, percent = evaluate(query_idx, retrieved_indices)
    
    save_path = os.path.join(RESULTS_DIR, f"query_{i+1}.png")
    visualize(query_idx, retrieved_indices, save_path)


end_time = time.time()

print("\n[INFO] Finished all queries.")
print(f"[INFO] Total runtime: {end_time - start_time:.2f} seconds")
print(f"[INFO] Results saved in: {RESULTS_DIR}")