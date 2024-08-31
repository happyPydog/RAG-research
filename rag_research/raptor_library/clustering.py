from typing import NamedTuple

import numpy as np
import tiktoken
import umap
from raptor_library.tree import Node
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# Config
CHUNK_SIZE = 100  # chunk size for the document
CHUNK_OVERLAP = 0  # overlap between two chunks
REDUCTION_COMPONENTS = 15  # number of components for UMAP
RANDOM_STATE = 123  # random state for GMM
MAX_LEVEL = 10  # maximum level for the clustering
MAX_TOKEN = 5000  # maximum tokens in a local cluster
MAX_CLUSTER_SIZE = 50  # maximum number of clusters
THRESHOLD = 0.1  # threshold for the clustering (depends on your document)

tokenizer = tiktoken.encoding_for_model("gpt-4o")


class ClusterResult(NamedTuple):
    clusters_array: np.ndarray
    n_clusters: int


class RAPTORClustering:
    def __init__(
        self,
        reduction_components: int,
        threshold: float,
        random_state: int,
        max_cluster_size: int,
    ):
        self.reduction_components = reduction_components
        self.threshold = threshold
        self.random_state = random_state
        self.max_cluster_size = max_cluster_size

    def __call__(self, embeddings: np.ndarray):
        embeddings = embeddings.copy()

        doc_size, _ = embeddings.shape

        total_cluster_count = 0
        total_clusters_labels = [
            np.array([], dtype=int) for _ in range(doc_size)
        ]  # [(doc_size, 0)]

        # * Global clustering
        # Use large n_neighbors for global clustering to capture the global structure.
        # In the author's implementation, they use n_neighbors = sqrt(embeddings_size - 1)
        global_n_neighbors = max(int((doc_size - 1) ** 0.5), 2)
        global_cluster_labels, global_n_clusters = self.cluster(
            embeddings=embeddings,
            n_components=self.reduction_components,
            n_neighbors=global_n_neighbors,
        )

        print(f"Number of global clusters: {global_n_clusters}")

        # * Local clustering
        for global_cluster_idx in tqdm(
            range(global_n_clusters), desc="Local Clustering..."
        ):
            # Extract embeddings for each global cluster
            embeddings_in_global_cluster = embeddings[
                np.array([global_cluster_idx in gc for gc in global_cluster_labels])
            ]

            # Handle the empty cluster
            # Sometimes, GMM may not find any cluster for the given threshold
            if len(embeddings_in_global_cluster) == 0:
                continue

            # Determine if local clustering is needed
            # If the number of embeddings in this global cluster is too small (<= reduction components + 1),
            # further clustering is unnecessary. Assign all embeddings to a single local cluster.
            if len(embeddings_in_global_cluster) <= self.reduction_components + 1:
                local_cluster_labels = [
                    np.array([0]) for _ in embeddings_in_global_cluster
                ]
                n_local_clusters = 1
            else:
                # * Perform local clustering
                # Use small n_neighbors for local clustering to capture the local structure.
                # In the author's implementation, they use n_neighbors = 10
                local_cluster_labels, n_local_clusters = self.cluster(
                    embeddings=embeddings_in_global_cluster,
                    n_components=self.reduction_components,
                    n_neighbors=10,
                )

            # * Update the total_clusters_label with local cluster assignments
            for local_cluster_idx in range(n_local_clusters):
                self.update_cluster_labels(
                    embeddings=embeddings,
                    embeddings_in_global_cluster=embeddings_in_global_cluster,
                    local_cluster_label=local_cluster_labels,
                    total_clusters_label=total_clusters_labels,
                    local_cluster_idx=local_cluster_idx,
                    total_cluster_count=total_cluster_count,
                )

            total_cluster_count += n_local_clusters

        print(f"Total number of clusters: {total_cluster_count}")

        return total_clusters_labels

    # @tenacity.retry(stop=(tenacity.stop_after_delay(20) | tenacity.stop_after_attempt(5)))
    def cluster(
        self, embeddings: np.ndarray, n_components: int, n_neighbors: int = 10
    ) -> ClusterResult:
        # Reduce the dimensionality by UMAP
        reduced_embeddings = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric="cosine",
        ).fit_transform(embeddings)

        # Find the optimal number of clusters
        n_clusters = self.find_cluster_count(reduced_embeddings)

        # Perform clustering
        gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        gmm.fit(reduced_embeddings)

        # Assign the cluster to each embedding
        probs = gmm.predict_proba(reduced_embeddings)

        # Given threshold, assign the cluster
        clusters_array = np.array(
            [np.where(prob >= self.threshold)[0] for prob in probs]
        )  # (doc_size, 1)

        return ClusterResult(clusters_array=clusters_array, n_clusters=n_clusters)

    def find_cluster_count(self, embeddings: np.ndarray) -> int:
        doc_size, _ = embeddings.shape
        # ? Determine the maximum number of clusters, ensuring it doesn't exceed the document count.
        # ? The number of clusters should not exceed the number of documents because:
        # ? 1. Each document could potentially be its own cluster, but you can't have more clusters than documents.
        # ? 2. Clustering aims to group similar documents, so having more clusters than documents is not meaningful.
        clusters_size = min(self.max_cluster_size, doc_size)
        n_clusters = np.arange(1, clusters_size)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=self.random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return n_clusters[np.argmin(bics)]

    def update_cluster_labels(
        self,
        embeddings: np.ndarray,
        embeddings_in_global_cluster: np.ndarray,
        local_cluster_label: np.ndarray,
        total_clusters_label: list[np.ndarray],
        local_cluster_idx: int,
        total_cluster_count: int,
    ) -> None:
        """Update the total_clusters_label with the new cluster index."""

        # Identify which embeddings belong to the current local cluster
        is_in_local_cluster = np.array(
            [local_cluster_idx in label for label in local_cluster_label]
        )
        embeddings_in_local_cluster = embeddings_in_global_cluster[is_in_local_cluster]

        # Find the indices of these embeddings in the original embeddings array
        expanded_embeddings = embeddings_in_local_cluster[:, np.newaxis]
        comparison_matrix = embeddings == expanded_embeddings
        matches = comparison_matrix.all(axis=-1)
        indices = np.where(np.any(matches, axis=0))[0]

        # Update the total_clusters_label with the new cluster index
        new_cluster_index = local_cluster_idx + total_cluster_count
        for idx in indices:
            current_label = total_clusters_label[idx]
            total_clusters_label[idx] = np.append(current_label, new_cluster_index)


def run_clustering(
    root_nodes: list[Node],
    reduction_components: int = REDUCTION_COMPONENTS,
    threshold: float = THRESHOLD,
    random_state: int = RANDOM_STATE,
    max_cluster_size: int = MAX_CLUSTER_SIZE,
    max_token: int = MAX_TOKEN,
    tokenizer: tiktoken.Encoding = tokenizer,
) -> list[Node]:
    """Runs the RAPTOR clustering algorithm on the given nodes."""
    model = RAPTORClustering(
        reduction_components=reduction_components,
        threshold=threshold,
        random_state=random_state,
        max_cluster_size=max_cluster_size,
    )

    # * Perform clustering
    embeddings = np.array([node.embeddings for node in root_nodes])
    cluster_labels = model(embeddings)
    n_cluster_labels = np.unique(np.concatenate(cluster_labels))

    result = []
    for label in n_cluster_labels:
        # Get the indices of the nodes that belong to this cluster
        indices = [i for i, cluster in enumerate(cluster_labels) if label in cluster]

        # Add the corresponding nodes to the node_clusters list
        cluster_nodes = [root_nodes[i] for i in indices]

        # Base case: if the cluster only has one node, do not attempt to recluster it
        if len(cluster_nodes) == 1:
            result.append(cluster_nodes)
            continue

        # Calculate the total length of the text in the nodes
        total_length = sum([len(tokenizer.encode(node.text)) for node in cluster_nodes])

        # If the total length exceeds the maximum allowed length, re-cluster this cluster
        if total_length > max_token:
            result.extend(run_clustering(nodes=cluster_nodes))
        else:
            result.append(cluster_nodes)

    return result
