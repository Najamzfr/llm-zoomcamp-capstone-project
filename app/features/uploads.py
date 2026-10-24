"""Upload your own paper (F11): session-only PDFs searched alongside the paper index.

PDF bytes are read in memory (PyMuPDF), split into ~800-token chunks with 100 tokens of overlap,
embedded with the app's fastembed model and kept in an in-memory Qdrant collection stored in
session_state. Nothing is written to disk; "New question" or the end of the session clears it.
When "Include my papers" is on, the agent fuses these chunks with its normal results via RRF.
"""

import ipaddress
import re
import socket
import uuid
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st

MAX_FILES = 3
MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 60
CHUNK_TOKENS = 800
CHUNK_OVERLAP = 100
SEARCH_K = 3
COLLECTION = "uploads"
USER_AGENT = "PaperScout/1.0 (reads a PDF a visitor linked)"


class UploadError(ValueError):
    """A user-facing reason a PDF couldn't be added."""


# ------------------------------------------------------------------ extraction and chunking
def extract_pages(data):
    """[(page number, text)] from PDF bytes, with friendly errors for bad, long or scanned files."""
    if len(data) > MAX_BYTES:
        raise UploadError("This PDF is larger than 20 MB. Try a smaller version.")
    if not data.startswith(b"%PDF"):
        raise UploadError("This file isn't a PDF.")
    import fitz

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as error:
        raise UploadError("This PDF couldn't be opened. It may be damaged or password-protected.") from error
    try:
        if document.needs_pass:
            raise UploadError("This PDF is password-protected.")
        if document.page_count > MAX_PAGES:
            raise UploadError(f"This PDF has {document.page_count} pages; the limit is about {MAX_PAGES}.")
        pages = [(number, " ".join(page.get_text("text").split())) for number, page in enumerate(document, start=1)]
    finally:
        document.close()
    if sum(len(text) for _, text in pages) < 200:
        raise UploadError("This PDF has no selectable text. Try a version exported from the original document "
                          "rather than a scan.")
    return pages


def chunk_pages(pages, name, tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    """~`tokens`-token chunks with `overlap`; each remembers the page it starts on."""
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    ids, page_of = [], []
    for number, text in pages:
        encoded = encoding.encode(text + " ")
        ids += encoded
        page_of += [number] * len(encoded)
    chunks, start = [], 0
    while start < len(ids):
        end = min(start + tokens, len(ids))
        text = encoding.decode(ids[start:end]).strip()
        if text:
            chunks.append({"file": name, "page": page_of[start], "text": text, "index": len(chunks) + 1})
        if end == len(ids):
            break
        start = end - overlap
    return chunks


# ------------------------------------------------------------------ fetching a linked PDF
def _public_host(hostname):
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(hostname, None)}
    except socket.gaierror as error:
        raise UploadError("That address couldn't be found. Check the link.") from error
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not ip.is_global:
            raise UploadError("That link points to a private network address.")


def fetch_pdf(url, max_redirects=3):
    """(file name, bytes) for a public http(s) link to a PDF; each redirect is re-checked."""
    for _ in range(max_redirects + 1):
        parsed = urlparse((url or "").strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise UploadError("Paste a full link that starts with https://")
        _public_host(parsed.hostname)
        try:
            response = requests.get(url, stream=True, timeout=20, allow_redirects=False, headers={"User-Agent": USER_AGENT})
        except requests.RequestException as error:
            raise UploadError("That link couldn't be reached. Try again or upload the file instead.") from error
        if response.is_redirect:
            url = urljoin(url, response.headers.get("location", ""))
            response.close()
            continue
        try:
            if response.status_code != 200:
                raise UploadError(f"That link returned an error ({response.status_code}).")
            data = b""
            for block in response.iter_content(64 * 1024):
                data += block
                if len(data) > MAX_BYTES:
                    raise UploadError("This PDF is larger than 20 MB. Try a smaller version.")
        finally:
            response.close()
        if not data.startswith(b"%PDF"):
            raise UploadError("That link doesn't lead to a PDF file. Use the direct link to the .pdf.")
        name = parsed.path.rstrip("/").rsplit("/", 1)[-1] or parsed.hostname
        return (name if name.lower().endswith(".pdf") else name + ".pdf"), data
    raise UploadError("That link redirects too many times.")


# ------------------------------------------------------------------ session index
def _store():
    return st.session_state.setdefault("ps_uploads", {"client": None, "files": []})


def files():
    return _store()["files"]


def clear():
    st.session_state["ps_uploads"] = {"client": None, "files": []}


def _embedder():
    from research_assistant.config import settings
    from research_assistant.embeddings import get_embedder

    return get_embedder(settings.embedding_model)


def add(name, data):
    """Index one PDF for this session; returns its page count."""
    store = _store()
    if len(store["files"]) >= MAX_FILES:
        raise UploadError(f"You can add up to {MAX_FILES} PDFs. Remove one first.")
    if any(item["name"] == name for item in store["files"]):
        raise UploadError("That file is already added.")
    pages = extract_pages(data)
    chunks = chunk_pages(pages, name)
    vectors = _embedder().embed_documents([chunk["text"] for chunk in chunks], batch_size=4)
    from qdrant_client import QdrantClient, models

    if store["client"] is None:
        store["client"] = QdrantClient(":memory:")
        store["client"].create_collection(COLLECTION, vectors_config=models.VectorParams(
            size=int(vectors.shape[1]), distance=models.Distance.COSINE))
    store["client"].upsert(COLLECTION, points=[
        models.PointStruct(id=str(uuid.uuid4()), vector=vector.tolist(), payload=chunk)
        for vector, chunk in zip(vectors, chunks)
    ])
    store["files"].append({"name": name, "pages": len(pages), "chunks": len(chunks)})
    return len(pages)


def remove(name):
    store = _store()
    store["files"] = [item for item in store["files"] if item["name"] != name]
    if store["client"] is not None:
        from qdrant_client import models

        store["client"].delete(COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="file", match=models.MatchValue(value=name))])))
    if not store["files"]:
        clear()


def _title(name):
    return re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)


def search(query, k=SEARCH_K, client=None):
    """Best chunks for `query` shaped like search results (the agent's extra_search hook)."""
    client = client or _store()["client"]
    if client is None:
        return []
    vector = _embedder().embed_query(query)
    hits = client.query_points(COLLECTION, query=vector.tolist(), limit=k).points
    return [{
        "paper_id": f"upload:{hit.payload['file']}:c{hit.payload['index']}",
        "title": _title(hit.payload["file"]),
        "text": hit.payload["text"],
        "category": None, "all_categories": "", "submitted_date": "", "authors": "",
        "arxiv_url": "", "pdf_url": "", "score": float(hit.score),
        "upload": True, "page": hit.payload["page"],
    } for hit in hits]


def searcher():
    """A search function bound to this session's index (safe to call from the agent)."""
    client = _store()["client"]
    return (lambda query, k=SEARCH_K: search(query, k, client)) if client is not None else None


# ------------------------------------------------------------------ UI
def _add_uploaded():
    state = st.session_state
    errors = []
    for item in state.get(f"upload_files_{state.get('upload_nonce', 0)}") or []:
        try:
            add(item.name, item.getvalue())
        except UploadError as error:
            errors.append(f"{item.name}: {error}")
    state["upload_errors"] = errors
    state["upload_nonce"] = state.get("upload_nonce", 0) + 1  # resets the uploader


def _add_link():
    state = st.session_state
    try:
        name, data = fetch_pdf(state.get("upload_url", ""))
        add(name, data)
        state["upload_errors"] = []
        state["upload_url"] = ""
    except UploadError as error:
        state["upload_errors"] = [str(error)]


def render_manager():
    """Contents of the 'Your papers' popover next to the composer."""
    state = st.session_state
    st.markdown("**Your papers**")
    st.caption(f"Up to {MAX_FILES} PDFs (20 MB, about {MAX_PAGES} pages each). They stay in this session only "
               "and are removed when you start a new question.")
    if len(files()) < MAX_FILES:
        st.file_uploader("Add PDFs", type=["pdf"], accept_multiple_files=True, key=f"upload_files_{state.get('upload_nonce', 0)}",
                         on_change=_add_uploaded)
        st.text_input("…or paste a link to a PDF", key="upload_url", placeholder="https://…/paper.pdf")
        st.button("Add link", key="upload_url_add", on_click=_add_link)
    for message in state.get("upload_errors") or []:
        st.error(message)
    for item in files():
        with st.container(horizontal=True, key=f"upload_row_{re.sub(r'[^A-Za-z0-9]', '_', item['name'])}"):
            st.markdown(f"{item['name']} · {item['pages']} pages")
            st.button("Remove", key=f"upload_remove_{re.sub(r'[^A-Za-z0-9]', '_', item['name'])}", type="tertiary",
                      on_click=remove, args=(item["name"],))
