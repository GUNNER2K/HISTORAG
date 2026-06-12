import os
import time
import h5py
import faiss
import random
import numpy as np
import matplotlib.pyplot as plt

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
# CREATE RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = Path(RESULTS_DIR) / "h1"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# ============================================================
# LOAD SINGLE H5
# ============================================================

def load_h5(path):

    with h5py.File(path, 'r') as f:

        features = f["features"][:]
        coords = f["coords"][:]

    return features, coords

# ============================================================
# DEMO MODE
# ============================================================

if not USE_DEMO == False:

    print("\n================================================")
    print("RUNNING DEMO MODE")
    print("================================================")

    features, coords = load_h5(H5_PATH)

# ============================================================
# FULL MODE
# ============================================================

else:

    print("\n================================================")
    print("RUNNING FULL MODE")
    print("================================================")

    all_h5_files = list(
        H5_ROOT.rglob("*.h5")
    )

    if MAX_WSIS is not None:

        all_h5_files = all_h5_files[:MAX_WSIS]

    all_features = []

    total_wsis = 0

    for h5_file in all_h5_files:

        try:

            print(f"\nLoading: {h5_file.name}")

            features_i, _ = load_h5(h5_file)

            all_features.append(
                features_i
            )

            total_wsis += 1

        except Exception as e:

            print(f"Skipping {h5_file.name}")
            print(e)

    features = np.vstack(
        all_features
    )

    print(f"\nLoaded WSIs: {total_wsis}")

# ============================================================
# FEATURE NORMALIZATION
# ============================================================

print("\nNormalizing features...")

features = features.astype(
    np.float32
)

features = features / np.linalg.norm(

    features,
    axis=1,
    keepdims=True

)

print("\nFinal feature matrix:", features.shape)

# ============================================================
# RETRIEVAL BASE CLASS
# ============================================================

class RetrievalEngine:

    def fit(self, X):
        pass

    def retrieve(self, q, k):
        pass

# ============================================================
# BRUTE FORCE
# ============================================================

class BruteForceRetrieval(RetrievalEngine):

    def fit(self, X):

        self.X = X

        print("\nBrute Force ready")

    def retrieve(self, q, k):

        sims = self.X @ q

        indices = np.argsort(
            -sims
        )[:k]

        return indices, sims[indices]

# ============================================================
# FAISS FLAT
# ============================================================

class FAISSFlatRetrieval(RetrievalEngine):

    def fit(self, X):

        d = X.shape[1]

        self.index = faiss.IndexFlatIP(d)

        self.index.add(X)

        print("\nFAISS Flat ready")

    def retrieve(self, q, k):

        scores, indices = self.index.search(

            q.reshape(1, -1),
            k

        )

        return indices[0], scores[0]

# ============================================================
# FAISS IVF
# ============================================================

class FAISSIVFRetrieval(RetrievalEngine):

    def fit(self, X):

        d = X.shape[1]

        nlist = 10

        quantizer = faiss.IndexFlatIP(d)

        self.index = faiss.IndexIVFFlat(

            quantizer,
            d,
            nlist,
            faiss.METRIC_INNER_PRODUCT

        )

        print("\nTraining IVF...")

        self.index.train(X)

        self.index.add(X)

        self.index.nprobe = 5

        print("\nFAISS IVF ready")

    def retrieve(self, q, k):

        scores, indices = self.index.search(

            q.reshape(1, -1),
            k

        )

        return indices[0], scores[0]

# ============================================================
# FAISS HNSW
# ============================================================

class FAISSHNSWRetrieval(RetrievalEngine):

    def fit(self, X):

        d = X.shape[1]

        self.index = faiss.IndexHNSWFlat(
            d,
            32
        )

        self.index.hnsw.efConstruction = 40

        self.index.add(X)

        print("\nFAISS HNSW ready")

    def retrieve(self, q, k):

        D, I = self.index.search(

            q.reshape(1, -1),
            k

        )

        return I[0], D[0]

# ============================================================
# RETRIEVAL METHODS
# ============================================================

retrievers = {

    "BruteForce":
    BruteForceRetrieval(),

    "FAISSFlat":
    FAISSFlatRetrieval(),

    "FAISSIVF":
    FAISSIVFRetrieval(),

    "FAISSHNSW":
    FAISSHNSWRetrieval()

}

# ============================================================
# RUN EXPERIMENTS
# ============================================================

latency_results = {}

for name, retriever in retrievers.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    retriever.fit(features)

    latencies = []

    # --------------------------------------------------------
    # RANDOM QUERIES
    # --------------------------------------------------------

    for _ in range(NUM_QUERIES):

        query_idx = np.random.randint(
            len(features)
        )

        query = features[query_idx]

        start = time.time()

        retrieved, scores = retriever.retrieve(
            query,
            TOP_K
        )

        latency = (

            time.time() - start

        ) * 1000

        latencies.append(latency)

    avg_latency = np.mean(
        latencies
    )

    latency_results[name] = avg_latency

    print(
        f"\nAverage Latency: "
        f"{avg_latency:.4f} ms"
    )

# ============================================================
# SORT FOR VISUALIZATION
# ============================================================

sorted_items = sorted(

    latency_results.items(),

    key=lambda x: x[1],

    reverse=True

)

methods = [x[0] for x in sorted_items]

latencies = [x[1] for x in sorted_items]

# ============================================================
# BARPLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    methods,
    latencies
)

for i, v in enumerate(latencies):

    plt.text(
        i,
        v,
        f"{v:.2f}",
        ha='center'
    )

plt.ylabel("Latency (ms)")

plt.xlabel("Retrieval Method")

plt.title("Retrieval Latency Comparison")

plt.tight_layout()

save_path = (

    RESULTS_DIR
    / "latency_comparison.png"

)

plt.savefig(
    save_path,
    dpi=300
)

plt.close()

# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = (

    RESULTS_DIR
    / "latency_summary.txt"

)

with open(summary_path, "w") as f:

    for method, latency in latency_results.items():

        f.write(
            f"{method}: "
            f"{latency:.4f} ms\n"
        )

# ============================================================
# FINAL PRINT
# ============================================================

print("\n================================================")
print("FINAL RESULTS")
print("================================================")

for method, latency in latency_results.items():

    print(
        f"{method}: "
        f"{latency:.4f} ms"
    )

print("\nSaved plot:")
print(save_path)
