import collections
import datetime
import functools
import itertools
import os
import re
import ssl
import tempfile
import urllib.error
import urllib.request
from typing import Any, NamedTuple

import arxiv
import fitz
import numpy as np
import tenacity
import tiktoken
import umap
from llama_index.core.node_parser.text import SentenceSplitter
from openai import OpenAI
from pydantic import BaseModel, Field, computed_field, field_validator
from raptor_library.utils import fetch_multiple_papers
from rich import print
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from tqdm.notebook import tqdm

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # replace by your openai api key

# Config
CHUNK_SIZE = 100  # chunk size for the document
CHUNK_OVERLAP = 0  # overlap between two chunks
REDUCTION_COMPONENT_SIZE = 30  # number of components for UMAP
RANDOM_STATE = 123  # random state for GMM
MAX_LAYERS = 10  # maximum level for the clustering
MAX_TOKEN = 5000  # maximum tokens in a local cluster
MAX_CLUSTER_SIZE = 20  # maximum number of clusters
THRESHOLD = 0.1  # threshold for the clustering (depends on your document)

llm = OpenAI(api_key=OPENAI_API_KEY)
tokenizer = tiktoken.encoding_for_model("gpt-4o")
embed_model = SentenceTransformer(
    "sentence-transformers/multi-qa-mpnet-base-cos-v1"
)  # vector size 768


class EmbeddingModel:

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    @functools.cache
    def __call__(self, text: str):
        return self.model.encode(text)


class Node(BaseModel):
    text: str
    layer: int
    children: set["Node"] = Field(default_factory=set)
    embeddings: np.ndarray | Any = Field(repr=False)

    class Config:
        arbitrary_types_allowed = True

    def add_child(self, child: "Node") -> None:
        self.children.add(child)

    def get_children(self) -> set["Node"]:
        return self.children

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_recur_children(self) -> set["Node"]:
        return self.children | {
            descendant
            for child in self.children
            for descendant in child.get_recur_children()
        }


def make_node(
    text: str, layer: int, embed_model: EmbeddingModel, children: set[Node]
) -> Node:
    embeddings = embed_model(text)
    return Node(text=text, layer=layer, children=children, embeddings=embeddings)


def make_leaf_nodes(texts: list[str], embed_model: EmbeddingModel) -> list[Node]:
    return [
        make_node(text=text, layer=0, children=set(), embed_model=embed_model)
        for text in tqdm(texts, desc="Creating embeddings", total=len(texts))
    ]


class Tree(BaseModel):
    layer_nodes_map: dict[int, list[Node]] = Field(
        default_factory=collections.defaultdict(list)
    )

    @computed_field
    @property
    def leaf_nodes(self) -> list[Node]:
        return self.layer_nodes_map[0]

    @field_validator("leaf_nodes")
    @classmethod
    def check_is_leaf(cls, v: list[Node]) -> list[Node]:
        for node in v:
            if not node.is_leaf():
                raise ValueError("All nodes must be leaf nodes.")
        return v

    def add_node(self, parent: Node, child: Node) -> None:
        parent.add_child(child)

    def traverse(self, nodes: list[Node] | None = None) -> list[str]:
        if nodes is None:
            nodes = self.leaf_nodes
        all_nodes = []
        for node in nodes:
            all_nodes.append(node.text)
            all_nodes.extend(self.traverse(list(node.get_children())))
        return all_nodes

    def find_node(self, node_text: str, nodes: list[Node] | None = None) -> Node | None:
        if nodes is None:
            nodes = self.leaf_nodes
        for node in nodes:
            if node.text == node_text:
                return node
            found_node = self.find_node(node_text, list(node.get_children()))
            if found_node:
                return found_node
        return None

    def get_leaves(self) -> list[Node]:
        """Retrieve all leaf nodes in the tree."""
        leaves = []

        def _find_nodes(nodes: list[Node]) -> None:
            for node in nodes:
                if node.is_leaf():
                    leaves.append(node)
                else:
                    _find_nodes(list(node.get_children()))

        _find_nodes(self.leaf_nodes)
        return leaves

    def get_nodes_at_layer(self, layer: int) -> list[Node]:
        """Retrieve all nodes at a specific layer."""
        nodes_at_layer = []

        def _find_nodes(nodes: list[Node], current_layer: int) -> None:
            for node in nodes:
                if current_layer == layer:
                    nodes_at_layer.append(node)
                if current_layer < layer:
                    _find_nodes(list(node.get_children()), current_layer + 1)

        _find_nodes(self.leaf_nodes, 0)
        return nodes_at_layer


class ClusterResult(NamedTuple):
    clusters_array: np.ndarray
    n_clusters: int


class RAPTORClustering:
    def __init__(
        self,
        reduction_component_size: int,
        threshold: float,
        random_state: int,
        max_cluster_size: int,
    ):
        self.reduction_component_size = reduction_component_size
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
            n_components=self.reduction_component_size,
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
            if len(embeddings_in_global_cluster) <= self.reduction_component_size + 1:
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
                    n_components=self.reduction_component_size,
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
    nodes: list[Node],
    reduction_component_size: int = REDUCTION_COMPONENT_SIZE,
    threshold: float = THRESHOLD,
    random_state: int = RANDOM_STATE,
    max_cluster_size: int = MAX_CLUSTER_SIZE,
    max_token: int = MAX_TOKEN,
    tokenizer: tiktoken.Encoding = tokenizer,
) -> list[Node]:
    """Runs the RAPTOR clustering algorithm on the given nodes."""
    model = RAPTORClustering(
        reduction_components=reduction_component_size,
        threshold=threshold,
        random_state=random_state,
        max_cluster_size=max_cluster_size,
    )

    # * Perform clustering
    embeddings = np.array([node.embeddings for node in nodes])
    cluster_labels = model(embeddings)
    n_cluster_labels = np.unique(np.concatenate(cluster_labels))

    result = []
    for label in n_cluster_labels:
        # Get the indices of the nodes that belong to this cluster
        indices = [i for i, cluster in enumerate(cluster_labels) if label in cluster]

        # Add the corresponding nodes to the node_clusters list
        cluster_nodes = [nodes[i] for i in indices]

        # Base case: if the cluster only has one node, do not attempt to re-cluster it
        if len(cluster_nodes) == 1:
            result.append(cluster_nodes)
            continue

        # Calculate the total length of the text in the nodes
        total_length = sum([len(tokenizer.encode(node.text)) for node in cluster_nodes])

        # If the total length exceeds the maximum allowed length, re-cluster this cluster
        if total_length > max_token:
            result.extend(run_clustering(root_nodes=cluster_nodes))
        else:
            result.append(cluster_nodes)

    return result


def summarize_cluster(llm: OpenAI, nodes: list[Node]) -> str:
    """Summarize the cluster of nodes using the given language model."""
    # Extract the context from the nodes
    context = ""
    for node in nodes:
        context += " ".join(node.text.splitlines())
        context += "\n\n"

    # Generate a summary using the language model
    nodes_prompt = "Write a summary of the following, including as many key details as possible: {context}:".format(
        context=context
    )
    response = llm.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": nodes_prompt},
        ],
    )

    return response.choices[0].message.content


def make_tree(
    texts: list[str],
    *,
    llm: OpenAI,
    embed_model: EmbeddingModel,
    reduction_component_size: int = REDUCTION_COMPONENT_SIZE,
    threshold: float = THRESHOLD,
    random_state: int = RANDOM_STATE,
    max_cluster_size: int = MAX_CLUSTER_SIZE,
    max_token: int = MAX_TOKEN,
    max_layers: int = MAX_LAYERS,
) -> Tree:
    """Make a tree from the given texts."""

    # Initialize the tree
    tree = Tree()

    # Construct the tree layer by layer
    curr_layer_nodes: list[Node] = []
    for layer in range(max_layers):
        if layer == 0:
            leaf_nodes = make_leaf_nodes(texts, embed_model)
            tree.layer_nodes_map[layer] = leaf_nodes
            continue

        # get last layer nodes
        last_nodes = tree.get_nodes_at_layer(layer - 1)

        # run clustering
        clusters = run_clustering(
            last_nodes,
            reduction_component_size=reduction_component_size,
            threshold=threshold,
            random_state=random_state,
            max_cluster_size=max_cluster_size,
            max_token=max_token,
            tokenizer=tokenizer,
        )

        # compose nodes based on the clusters
        for cluster in clusters:
            summary = summarize_cluster(llm, cluster)
            nodes = make_node(
                text=summary,
                layer=layer,
                embed_model=embed_model,
                children=set(cluster),
            )
            curr_layer_nodes.append(nodes)

        # update the tree
        tree.layer_nodes_map[layer] = curr_layer_nodes

    return tree


class RAPTORRetriever:

    def __init__(self, tree: Tree):
        self.tree = tree


def main():
    # Download document resources
    urls = [
        "https://arxiv.org/abs/2401.18059",
        "https://arxiv.org/abs/2307.03172",
    ]
    papers = fetch_multiple_papers(urls)

    # Split text into sentences
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = itertools.chain.from_iterable(
        splitter.split_text(paper.text) for paper in papers
    )

    llm = OpenAI(api_key=OPENAI_API_KEY)

    # Make tree
    tree = make_tree(
        texts,
        llm=llm,
        embed_model=embed_model,
        reduction_component_size=REDUCTION_COMPONENT_SIZE,
        threshold=THRESHOLD,
        random_state=RANDOM_STATE,
        max_cluster_size=MAX_CLUSTER_SIZE,
        max_token=MAX_TOKEN,
        max_layers=MAX_LAYERS,
    )

    # Retriever


if __name__ == "__main__":
    main()
