import os
import time
import h5py
import faiss
import numpy as np
import openslide
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

tumor_feat_path = "/home/woody/iwi5/iwi5411h/BIMAP/patches_tumor/uni/20x_256px_0px_overlap/features_uni_v1/TumorCenter_CD3_block1.h5"

inv_feat_path = "/home/woody/iwi5/iwi5411h/BIMAP/patches_inv/uni/20x_256px_0px_overlap/features_uni_v1/InvasionFront_CD3_block1.h5"

tumor_wsi_path = "/home/woody/iwi5/iwi5411h/BIMAP/data/TumorCenter_CD3_block1.svs"

inv_wsi_path = "/home/woody/iwi5/iwi5411h/BIMAP/data/InvasionFront_CD3_block1.svs"

RESULTS_DIR = "/home/woody/iwi5/iwi5411h/BIMAP/results"

PATCH_SIZE = 256
LEVEL = 0
NUM_QUERIES = 20

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD H5
# ============================================================

def load_h5(path,name):

    print(f"\nLoading {name}")

    with h5py.File(path,'r') as f:

        features=f["features"][:]
        coords=f["coords"][:]

    print(features.shape)

    return features,coords


X_T,C_T=load_h5(
    tumor_feat_path,
    "Tumor"
)

X_I,C_I=load_h5(
    inv_feat_path,
    "Invasion"
)


# ============================================================
# DATA PREP
# ============================================================

X=np.vstack([X_T,X_I])

coords=np.vstack([
    C_T,
    C_I
])

slide_ids=np.concatenate([

    np.zeros(
        len(X_T)
    ),

    np.ones(
        len(X_I)
    )

])

# temporary labels
labels=slide_ids.copy()

print(
    "\nTotal patches:",
    len(X)
)

print(
    "Feature dim:",
    X.shape[1]
)


# ============================================================
# NORMALIZATION
# ============================================================

X=X/np.linalg.norm(
    X,
    axis=1,
    keepdims=True
)

X=X.astype(
    np.float32
)


# ============================================================
# LOAD WSI
# ============================================================

slide_T=openslide.OpenSlide(
    tumor_wsi_path
)

slide_I=openslide.OpenSlide(
    inv_wsi_path
)


# ============================================================
# PATCH EXTRACTION
# ============================================================

def get_patch(idx):

    x,y=coords[idx]

    slide=(

        slide_T

        if slide_ids[idx]==0

        else slide_I

    )

    patch=slide.read_region(

        (int(x),int(y)),
        LEVEL,
        (PATCH_SIZE,PATCH_SIZE)

    ).convert("RGB")

    return patch


# ============================================================
# RETRIEVAL BASE
# ============================================================

class RetrievalEngine:

    def fit(self,X):
        pass

    def retrieve(self,q,k):
        pass


# ============================================================
# BRUTE FORCE
# ============================================================

class BruteForceRetrieval(
    RetrievalEngine
):

    def fit(self,X):

        self.X=X

        print(
            "\nBrute Force ready"
        )

    def retrieve(
            self,
            q,
            k
    ):

        sims=self.X@q

        indices=np.argsort(
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

    def fit(self,X):

        d=X.shape[1]

        self.index=faiss.IndexFlatIP(
            d
        )

        self.index.add(X)

        print(
            "\nFAISS Flat ready"
        )


    def retrieve(
            self,
            q,
            k
    ):

        scores,indices=(
            self.index.search(
                q.reshape(
                    1,-1
                ),
                k
            )
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

    def fit(self,X):

        d=X.shape[1]

        nlist=100

        quantizer=faiss.IndexFlatIP(
            d
        )

        self.index=(
            faiss.IndexIVFFlat(

                quantizer,
                d,
                nlist,
                faiss.METRIC_INNER_PRODUCT

            )
        )

        print(
            "\nTraining IVF..."
        )

        self.index.train(X)

        self.index.add(X)

        self.index.nprobe=10

        print(
            "\nFAISS IVF ready"
        )


    def retrieve(
            self,
            q,
            k
    ):

        scores,indices=(
            self.index.search(
                q.reshape(
                    1,-1
                ),
                k
            )
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

    def fit(self,X):

        d=X.shape[1]

        self.index=(
            faiss.IndexHNSWFlat(
                d,
                32
            )
        )

        self.index.add(
            X
        )

        print(
            "\nFAISS HNSW ready"
        )


    def retrieve(
            self,
            q,
            k
    ):

        D,I=self.index.search(

            q.reshape(
                1,-1
            ),

            k

        )

        return (
            I[0],
            D[0]
        )


# ============================================================
# METRICS
# ============================================================

def precision_at_k(
        retrieved,
        query_label,
        labels
):

    relevant=sum(

        labels[idx]==query_label

        for idx in retrieved

    )

    return relevant/len(
        retrieved
    )


def recall_at_k(
        retrieved,
        query_label,
        labels
):

    total=np.sum(

        labels==query_label

    )

    relevant=sum(

        labels[idx]==query_label

        for idx in retrieved

    )

    return relevant/total


def average_precision(
        retrieved,
        query_label,
        labels
):

    precisions=[]

    relevant=0


    for rank,idx in enumerate(

            retrieved,
            start=1

    ):

        if labels[idx]==query_label:

            relevant+=1

            precisions.append(
                relevant/rank
            )


    if len(
            precisions
    )==0:

        return 0


    return np.mean(
        precisions
    )


# ============================================================
# VISUALIZATION
# ============================================================

def visualize(
        query_idx,
        retrieved,
        save_path
):

    plt.figure(
        figsize=(14,5)
    )

    plt.subplot(
        2,
        len(retrieved)+1,
        1
    )

    plt.imshow(
        get_patch(
            query_idx
        )
    )

    plt.title(
        "Query"
    )

    plt.axis(
        "off"
    )


    for i,idx in enumerate(
            retrieved
    ):

        plt.subplot(
            2,
            len(retrieved)+1,
            i+2
        )

        plt.imshow(
            get_patch(idx)
        )

        plt.axis(
            "off"
        )


    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


# ============================================================
# RETRIEVAL METHODS
# ============================================================

retrievers={

    "brute_force":
        BruteForceRetrieval(),

    "faiss_flat":
        FAISSFlatRetrieval(),

    "faiss_ivf":
        FAISSIVFRetrieval(),

    "faiss_hnsw":
        FAISSHNSWRetrieval()

}


# ============================================================
# EXPERIMENTS
# ============================================================

latency_results={}


for name,retriever in retrievers.items():

    print("\n"+"="*50)
    print(name)
    print("="*50)

    retriever.fit(X)

    results=[]


    for q in range(NUM_QUERIES):

        query_idx=np.random.randint(
            len(X)
        )

        query=X[
            query_idx
        ]

        start=time.time()

        retrieved,scores=(

            retriever.retrieve(
                query,
                11
            )

        )

        latency=(
            time.time()-start
        )*1000


        mask=(
            retrieved!=query_idx
        )

        retrieved=(
            retrieved[mask][:10]
        )


        p5=precision_at_k(
            retrieved[:5],
            labels[query_idx],
            labels
        )

        p10=precision_at_k(
            retrieved,
            labels[query_idx],
            labels
        )

        r5=recall_at_k(
            retrieved[:5],
            labels[query_idx],
            labels
        )

        r10=recall_at_k(
            retrieved,
            labels[query_idx],
            labels
        )

        ap=average_precision(
            retrieved,
            labels[query_idx],
            labels
        )


        results.append({

            "P@5":p5,
            "P@10":p10,
            "R@5":r5,
            "R@10":r10,
            "AP":ap,
            "latency":latency

        })


        if q<3:

            save_path=os.path.join(

                RESULTS_DIR,
                f"{name}_query_{q+1}.png"

            )

            visualize(

                query_idx,
                retrieved,
                save_path

            )


    latency_results[name]=np.mean(

        [r["latency"]
         for r in results]

    )


    print("\nFINAL RESULTS")

    for metric in [

        "P@5",
        "P@10",
        "R@5",
        "R@10",
        "AP",
        "latency"

    ]:

        value=np.mean([

            r[metric]
            for r in results

        ])

        print(
            f"{metric}: {value:.4f}"
        )


# ============================================================
# SAVE LATENCY PLOT
# ============================================================

plt.figure(
    figsize=(8,5)
)

plt.bar(

    latency_results.keys(),
    latency_results.values()

)

plt.ylabel(
    "Latency (ms)"
)

plt.title(
    "Retrieval Method Comparison"
)

for i,v in enumerate(

    latency_results.values()

):

    plt.text(

        i,
        v,
        f"{v:.2f}",
        ha='center'

    )

plt.tight_layout()

save_path=os.path.join(

    RESULTS_DIR,
    "latency_comparison.png"

)

plt.savefig(
    save_path,
    dpi=300
)

plt.close()

print(
    f"\nSaved plot: {save_path}"
)