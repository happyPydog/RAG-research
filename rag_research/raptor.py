import itertools
import datetime
import os
import re
import ssl
import tempfile

import typing as t
import numpy as np

from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from tqdm.notebook import tqdm

from llama_index.core.node_parser.text import SentenceSplitter

import arxiv
import fitz
import tiktoken
import umap
import tenacity

import urllib.error
import urllib.request

from .raptor_library.utils import fetch_multiple_papers


class ArXivPaper(BaseModel):
    title: str = Field(..., description="Title of the paper")
    text: str = Field(..., description="Text content of the paper")
    url: str = Field(..., description="URL of the paper on ArXiv")


def fetch_single_paper(url: str) -> ArXivPaper:
    """Fetches and downloads an ArXiv paper from the given URL and returns its ArXivPaper object."""
    # Ensure SSL certificate is verified
    try:
        urllib.request.urlopen(url)
    except (ssl.SSLCertVerificationError, urllib.error.URLError):
        ssl._create_default_https_context = ssl._create_unverified_context

    # Extract ArXiv ID from the URL
    arxiv_match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
    if not arxiv_match:
        raise ValueError(
            f"Invalid ArXiv URL: {url}. Expected URL should contain ArXiv ID."
        )
    arxiv_id = arxiv_match.group(1)

    # Fetch paper using ArXiv ID
    client = arxiv.Client()
    search_query = arxiv.Search(id_list=[arxiv_id])
    search_result = client.results(search_query)

    try:
        paper = next(search_result)
    except StopIteration:
        raise ValueError(f"Paper not found for ArXiv ID: {arxiv_id}")

    # Download PDF and extract text content
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = paper.download_pdf(dirpath=temp_dir)
        doc = fitz.open(pdf_path)
        text = "".join([page.get_text() for page in doc])

    return ArXivPaper(title=paper.title, text=text, url=url)


class Node(BaseModel):
    text: str
    children: set["Node"] = Field(default_factory=set)
    embedding: t.Any | None = Field(default=None)

    @field_validator("embedding", mode="before")
    def to_numpy(cls, v: t.Any):
        if v is not None:
            return np.array(v)
        return v

    def add_child(self, child: "Node") -> None:
        self.children.add(child)

    def get_children(self) -> set["Node"]:
        return self.children

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class Tree(BaseModel):
    root_nodes: list[Node] = Field(default_factory=list)

    def add_node(self, parent: Node, child: Node) -> None:
        parent.add_child(child)

    def traverse(self, nodes: list[Node] | None = None) -> list[str]:
        if nodes is None:
            nodes = self.root_nodes
        all_nodes = []
        for node in nodes:
            all_nodes.append(node.text)
            all_nodes.extend(self.traverse(list(node.get_children())))
        return all_nodes

    def find_node(self, node_text: str, nodes: list[Node] | None = None) -> Node | None:
        if nodes is None:
            nodes = self.root_nodes
        for node in nodes:
            if node.text == node_text:
                return node
            found_node = self.find_node(node_text, list(node.get_children()))
            if found_node:
                return found_node
        return None

    def get_leaves(self) -> list[Node]: ...


def make_node(): ...


def make_tree(): ...


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # replace by your openai api key

# Config
CHUNK_SIZE = 100  # chunk size for the document
CHUNK_OVERLAP = 0  # overlap between two chunks
REDUCTION_COMPONENTS = 15  # number of components for UMAP
RANDOM_STATE = 123  # random state for GMM
MAX_LEVEL = 10  # maximum level for the clustering
MAX_TOKEN = 5000  # maximum tokens in a local cluster
MAX_CLUSTER_SIZE = 50  # maximum number of clusters
THRESHOLD = 0.1  # threshold for the clustering (depends on your document)

llm = OpenAI(api_key=OPENAI_API_KEY)
tokenizer = tiktoken.encoding_for_model("gpt-4o")


def main():
    # Download document resources
    urls = [
        "https://arxiv.org/pdf/2104.14764.pdf",
        "https://arxiv.org/pdf/2104.14764.pdf",
    ]
    papers = fetch_multiple_papers(urls)


if __name__ == "__main__":
    ...
