"""Safe PDF download, caching, page-aware chunking, and local chunk retrieval."""

import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import fitz
import numpy as np
import requests
import tiktoken
from .config import settings
from .embeddings import get_embedder
from .tracing import traceable
from .paths import PAPER_CACHE_DIR

def _summarize_retrieval(outputs):
    """Trace chunk ids, pages and scores rather than the full PDF text."""
    if isinstance(outputs, dict):
        outputs = outputs.get("output", outputs)
    chunks, _context, downloaded_ids = outputs
    return {
        "chunks": [
            {key: chunk.get(key) for key in ("chunk_id", "page", "score")}
            for chunk in chunks
        ],
        "downloaded_paper_ids": downloaded_ids,
    }


CACHE_DIR = PAPER_CACHE_DIR
PDF_DIR = CACHE_DIR / "pdfs"
CHUNK_DIR = CACHE_DIR / "chunks"
MAX_PAPERS = settings.max_papers
MAX_PDF_BYTES = settings.max_pdf_bytes
MAX_PDF_PAGES = settings.max_pdf_pages
MAX_CHUNKS_PER_PDF = settings.max_chunks_per_pdf
DOWNLOAD_TIMEOUT = settings.download_timeout_seconds
DOWNLOAD_DELAY_SECONDS = settings.download_delay_seconds
CHUNK_TOKENS = settings.chunk_tokens
CHUNK_OVERLAP = settings.chunk_overlap
MAX_CHUNKS = settings.max_chunks
MAX_CHUNKS_PER_PAPER = settings.max_chunks_per_paper
CHUNK_EMBED_BATCH_SIZE = settings.chunk_embed_batch_size
MAX_FULL_TEXT_TOKENS = settings.max_full_text_tokens
EMBEDDING_MODEL = settings.embedding_model
USER_AGENT = "ResearchPaperRAG/1.0 (educational project)"


class PaperTools:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.pdf_dir = self.cache_dir / "pdfs"
        self.chunk_dir = self.cache_dir / "chunks"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.model = None
        self.embedding_cache = {}

    def pdf_url(self, paper):
        url = (paper.get("pdf_url") or "").strip()
        if not url:
            abstract_url = (paper.get("arxiv_url") or "").strip()
            if "/abs/" in abstract_url:
                url = abstract_url.replace("/abs/", "/pdf/") + ".pdf"
            else:
                paper_id = paper["paper_id"]
                if not re.fullmatch(r"[\w.\-/]+", paper_id):
                    raise ValueError(f"Invalid arXiv paper ID: {paper_id}")
                url = f"https://arxiv.org/pdf/{paper_id}.pdf"

        return self._validate_arxiv_url(url)

    @staticmethod
    def _validate_arxiv_url(url):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("PDF URL must use HTTP or HTTPS.")
        if parsed.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            raise ValueError("Only arxiv.org PDF downloads are allowed.")
        return parsed._replace(scheme="https").geturl()

    def download(self, paper):
        destination = self.pdf_dir / f"{self._safe_id(paper['paper_id'])}.pdf"
        if destination.exists():
            with destination.open("rb") as file:
                signature = file.read(4)
            valid_cache = (
                4 < destination.stat().st_size <= MAX_PDF_BYTES and signature == b"%PDF"
            )
            if valid_cache:
                return destination, False
            destination.unlink(missing_ok=True)

        response = requests.get(
            self.pdf_url(paper),
            headers={"User-Agent": USER_AGENT},
            timeout=DOWNLOAD_TIMEOUT,
            stream=True,
        )
        temporary = destination.with_suffix(".pdf.part")
        try:
            response.raise_for_status()
            self._validate_arxiv_url(response.url)
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and "octet-stream" not in content_type:
                raise ValueError(
                    f"Expected a PDF but received {content_type or 'unknown content'}."
                )
            content_length = int(response.headers.get("content-length", 0) or 0)
            if content_length > MAX_PDF_BYTES:
                raise ValueError(
                    f"PDF exceeds the {MAX_PDF_BYTES // 1024**2} MB limit."
                )

            size = 0
            with temporary.open("wb") as file:
                for block in response.iter_content(chunk_size=64 * 1024):
                    if not block:
                        continue
                    size += len(block)
                    if size > MAX_PDF_BYTES:
                        raise ValueError(
                            f"PDF exceeds the {MAX_PDF_BYTES // 1024**2} MB limit."
                        )
                    file.write(block)
            with temporary.open("rb") as file:
                signature = file.read(4)
            if signature != b"%PDF":
                raise ValueError("Downloaded file is not a valid PDF.")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            response.close()
        return destination, True

    def chunks(self, paper):
        cache_path = self.chunk_dir / f"{self._safe_id(paper['paper_id'])}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                required = {"chunk_id", "paper_id", "title", "page", "text"}
                if (
                    isinstance(cached, list)
                    and cached
                    and all(
                        isinstance(chunk, dict) and required <= chunk.keys()
                        for chunk in cached
                    )
                ):
                    return cached, False
            except (json.JSONDecodeError, OSError):
                pass
            cache_path.unlink(missing_ok=True)

        pdf_path, downloaded = self.download(paper)
        chunks = self._extract_chunks(pdf_path, paper)
        cache_path.write_text(
            json.dumps(chunks, ensure_ascii=False),
            encoding="utf-8",
        )
        return chunks, downloaded

    @traceable(
        name="retrieve_full_text",
        run_type="tool",
        process_inputs=lambda inputs: {
            "query": inputs.get("query"),
            "paper_ids": [paper.get("paper_id") for paper in inputs.get("papers") or []],
        },
        process_outputs=_summarize_retrieval,
    )
    def retrieve(self, query, papers):
        selected = list(papers)[:MAX_PAPERS]
        all_chunks = []
        downloaded_ids = []
        for paper in selected:
            chunks, downloaded = self.chunks(paper)
            all_chunks.extend(chunks)
            if downloaded:
                downloaded_ids.append(paper["paper_id"])
                time.sleep(DOWNLOAD_DELAY_SECONDS)

        if not all_chunks:
            return [], "", downloaded_ids

        if self.model is None:
            self.model = get_embedder(EMBEDDING_MODEL)
        texts = [chunk["text"] for chunk in all_chunks]
        cache_key = tuple(chunk["chunk_id"] for chunk in all_chunks)
        embeddings = self.embedding_cache.get(cache_key)
        if embeddings is None:
            embeddings = self.model.embed_documents(texts, batch_size=CHUNK_EMBED_BATCH_SIZE)
            self.embedding_cache[cache_key] = embeddings

        query_vector = self.model.embed_query(query)
        scores = embeddings @ query_vector
        order = np.argsort(scores)[::-1]

        # Excerpts are labelled with their paper's source number (search rank), so
        # citations in full-text answers point at the same sources as abstract answers.
        source_numbers = {
            paper["paper_id"]: paper.get("rank", position)
            for position, paper in enumerate(selected, start=1)
        }
        retrieved = []
        per_paper = {}
        for index in order:
            chunk = dict(all_chunks[index])
            paper_id = chunk["paper_id"]
            if per_paper.get(paper_id, 0) >= MAX_CHUNKS_PER_PAPER:
                continue
            chunk["score"] = float(scores[index])
            chunk["source"] = source_numbers.get(paper_id)
            retrieved.append(chunk)
            per_paper[paper_id] = per_paper.get(paper_id, 0) + 1
            if len(retrieved) == MAX_CHUNKS:
                break

        context = self._limited_context(retrieved)
        return retrieved, context, downloaded_ids

    def _extract_chunks(self, pdf_path, paper):
        chunks = []
        document = fitz.open(pdf_path)
        try:
            for page_number, page in enumerate(document[:MAX_PDF_PAGES], start=1):
                text = " ".join(page.get_text("text").split())
                tokens = self.encoding.encode(text)
                start = 0
                number = 1
                while start < len(tokens):
                    end = min(start + CHUNK_TOKENS, len(tokens))
                    chunk_text = self.encoding.decode(tokens[start:end]).strip()
                    if chunk_text:
                        chunks.append(
                            {
                                "chunk_id": f"{paper['paper_id']}-p{page_number}-c{number}",
                                "paper_id": paper["paper_id"],
                                "title": paper["title"],
                                "page": page_number,
                                "text": chunk_text,
                            }
                        )
                        if len(chunks) >= MAX_CHUNKS_PER_PDF:
                            return chunks
                    if end == len(tokens):
                        break
                    start = end - CHUNK_OVERLAP
                    number += 1
        finally:
            document.close()
        if not chunks:
            raise ValueError(f"No extractable text found in {pdf_path.name}.")
        return chunks

    def _limited_context(self, chunks):
        parts = []
        remaining = MAX_FULL_TEXT_TOKENS
        for number, chunk in enumerate(chunks, start=1):
            separator = "\n\n" if parts else ""
            separator_tokens = self.encoding.encode(separator)
            label = chunk.get("source") or number
            header = (
                f"[{label}] {chunk['title']} "
                f"(paper_id: {chunk['paper_id']}; page: {chunk['page']})\n"
            )
            header_tokens = self.encoding.encode(header)
            fixed_tokens = len(separator_tokens) + len(header_tokens)
            if fixed_tokens >= remaining:
                break
            text_tokens = self.encoding.encode(chunk["text"])
            allowed = remaining - fixed_tokens
            text = self.encoding.decode(text_tokens[:allowed])
            parts.append(separator + header + text)
            remaining -= fixed_tokens + min(len(text_tokens), allowed)
            if remaining <= 0:
                break
        context = "".join(parts)
        assert len(self.encoding.encode(context)) <= MAX_FULL_TEXT_TOKENS
        return context

    @staticmethod
    def _safe_id(paper_id):
        return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)
