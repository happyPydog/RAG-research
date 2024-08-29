import re
import ssl
import tempfile

import arxiv
import fitz
import urllib.error
import urllib.request

from pydantic import BaseModel, Field


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


def fetch_multiple_papers(urls: list[str]) -> list[ArXivPaper]:
    """Fetches and downloads multiple ArXiv papers from the given URLs and returns a list of ArXivPaper objects."""
    return [fetch_single_paper(url) for url in urls]


def prepare_documents(papers: list[ArXivPaper]) -> list[Node]:
    """Prepares a list of documents from ArXivPaper."""
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    sentences = itertools.chain.from_iterable(
        splitter.split_text(paper.text) for paper in papers
    )
    return [Node(text=sentence) for sentence in sentences]
