import os
import time
import h5py
import faiss
import random
import numpy as np
import matplotlib.pyplot as plt

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

RESULTS_DIR = (
    ROOT_DIR
    / "results"
    / "h1"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

NUM_QUERIES = 20

# ============================================================
# LOAD H5
# ============================================================

print("\nLoading demo H5...")

with h5py.File(H5_PATH, 'r') as f:

    features = f["features"][:]
    coords = f["coords"][:]

print("Features:", features.shape)
print("Coords:", coords.shape)

# ============================================================
# NORMALIZATION
# ============================================================

features = features.astype(
    np.float32
)

features = features / np.linalg.norm(

    features,
    axis=1,
    keepdims=True

)

# ============================================================
# RETRIEVAL BASE
# ============================================================

class RetrievalEngine:

    def fit(self, X):
        pass

    def retrieve(self, q, k):
        pass

# ============================================================
# BRUTE FORCE
# ============================================================

class BruteForceRetrieval(
    RetrievalEngine
):

    def fit(self, X):

        self.X = X

        print(
            "\nBrute Force ready"
        )

    def retrieve(
        self,
        q,
        k
    ):

        sims = self.X @ q

        indices = np.argsort(
            -sims
        )[:k]

        return (
            indices,
            sims[indices]
        )

# ============================================================
# FAISS FLAT
# ============================================================

class FAISSFlatRetrieval(
    RetrievalEngine
):

    def fit(self, X):

        d = X.shape[1]

        self.index = faiss.IndexFlatIP(d)

        self.index.add(X)

        print(
            "\nFAISS Flat ready"
        )

    def retrieve(
        self,
        q,
        k
    ):

        scores, indices = self.index.search(

            q.reshape(1, -1),
            k

        )

        return (
            indices[0],
            scores[0]
        )

# ============================================================
# FAISS IVF
# ============================================================

class FAISSIVFRetrieval(
    RetrievalEngine
):

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

        print(
            "\nTraining IVF..."
        )

        self.index.train(X)

        self.index.add(X)

        self.index.nprobe = 5

        print(
            "\nFAISS IVF ready"
        )

    def retrieve(
        self,
        q,
        k
    ):

        scores, indices = self.index.search(

            q.reshape(1, -1),
            k

        )

        return (
            indices[0],
            scores[0]
        )

# ============================================================
# FAISS HNSW
# ============================================================

class FAISSHNSWRetrieval(
    RetrievalEngine
):

    def fit(self, X):

        d = X.shape[1]

        self.index = faiss.IndexHNSWFlat(
            d,
            32
        )

        self.index.hnsw.efConstruction = 40

        self.index.add(X)

        print(
            "\nFAISS HNSW ready"
        )

    def retrieve(
        self,
        q,
        k
    ):

        D, I = self.index.search(

            q.reshape(1, -1),
            k

        )

        return (
            I[0],
            D[0]
        )

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
# EXPERIMENTS
# ============================================================

latency_results = {}

for name, retriever in retrievers.items():

    print("\n" + "="*50)
    print(name)
    print("="*50)

    retriever.fit(features)

    latencies = []

    # --------------------------------------------------------
    # MULTIPLE RANDOM QUERIES
    # --------------------------------------------------------

    for _ in range(NUM_QUERIES):

        query_idx = np.random.randint(
            len(features)
        )

        query = features[
            query_idx
        ]

        start = time.time()

        retrieved, scores = retriever.retrieve(
            query,
            10
        )

        latency = (
            time.time() - start
        ) * 1000

        latencies.append(
            latency
        )

    avg_latency = np.mean(
        latencies
    )

    latency_results[name] = avg_latency

    print(
        f"\nAverage Latency: "
        f"{avg_latency:.4f} ms"
    )

# ============================================================
# SORT BY LATENCY
# ============================================================

sorted_items = sorted(

    latency_results.items(),

    key=lambda x: x[1],

    reverse=True

)

methods = [
    x[0]
    for x in sorted_items
]

latencies = [
    x[1]
    for x in sorted_items
]

# ============================================================
# LATENCY BARPLOT
# ============================================================

plt.figure(figsize=(8,5))

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

plt.title(
    "Latency Comparison"
)

plt.tight_layout()

save_path = os.path.join(

    RESULTS_DIR,
    "latency_comparison.png"

)

plt.savefig(
    save_path,
    dpi=300
)

plt.close()

# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(

    RESULTS_DIR,
    "latency_summary.txt"

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
