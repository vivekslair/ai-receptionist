"""
Maya Knowledge Base Ingestor
============================
Builds kb_index.json from documents and URLs so Maya can answer
questions accurately about your business.

Usage
-----
# Index everything in kb/ (PDFs, TXT, MD) + URLs in kb/urls.txt
python ingest.py

# Also crawl an extra URL on the fly
python ingest.py --url https://example.com/about

# Index a single file directly
python ingest.py --file /path/to/brochure.pdf

Output
------
kb_index.json — loaded automatically when the server starts.
Restart the server (or it hot-reloads via uvicorn --reload) after ingesting.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

KB_DIR   = Path("kb")
INDEX    = Path("kb_index.json")
CHUNK_SZ = 500   # target chars per chunk
OVERLAP  = 80    # chars of overlap between consecutive chunks


# ─────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────
# Chunking — paragraph-aware with character overlap
# ─────────────────────────────────────────────────────────────────
def chunk_text(text: str, source: str) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks: list[dict] = []
    buf = ""

    for para in paragraphs:
        if len(buf) + len(para) + 2 > CHUNK_SZ and buf:
            chunks.append(_make_chunk(buf, source, len(chunks)))
            buf = buf[-OVERLAP:] + "\n\n" + para  # carry overlap forward
        else:
            buf = (buf + "\n\n" + para).strip() if buf else para

    if buf.strip():
        chunks.append(_make_chunk(buf, source, len(chunks)))

    return chunks


def _make_chunk(text: str, source: str, idx: int) -> dict:
    return {
        "id":     hashlib.md5(f"{source}:{idx}:{text[:40]}".encode()).hexdigest()[:10],
        "source": source,
        "text":   text.strip(),
    }


# ─────────────────────────────────────────────────────────────────
# Readers
# ─────────────────────────────────────────────────────────────────
def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        log.error("pypdf not installed — run: pip install pypdf")
        return ""

    reader = PdfReader(str(path))
    pages  = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def read_url(url: str) -> str:
    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError:
        log.error("httpx / beautifulsoup4 not installed — run: pip install httpx beautifulsoup4")
        return ""

    log.info("Fetching %s …", url)
    try:
        r = httpx.get(url, timeout=20, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (Maya KB Ingestor)"})
        r.raise_for_status()
    except Exception as exc:
        log.error("Failed to fetch %s: %s", url, exc)
        return ""

    soup = BeautifulSoup(r.text, "html.parser")

    # Extract meta description and title first (often best summary for product sites)
    extra_parts: list[str] = []
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        extra_parts.append("Page title: " + title_tag.get_text(strip=True))

    for meta in soup.find_all("meta", attrs={"name": True, "content": True}):
        if meta["name"].lower() in ("description", "keywords"):
            extra_parts.append(f"Meta {meta['name']}: {meta['content'].strip()}")

    # Collect alt text from images (product images often have descriptive alts)
    alt_texts = []
    for img in soup.find_all("img", alt=True):
        alt = img["alt"].strip()
        if len(alt) > 5 and not alt.lower().startswith("icon"):
            alt_texts.append(alt)
    if alt_texts:
        extra_parts.append("Product/image descriptions: " + " | ".join(alt_texts[:30]))

    # Remove boilerplate tags
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "iframe", "noscript", "svg"]):
        tag.decompose()

    # Prefer main content areas
    main = soup.find("main") or soup.find("article") or soup.find("body") or soup

    # Pull structured list content explicitly (good for product lists)
    list_items: list[str] = []
    for li in main.find_all(["li", "dt", "dd"]):
        txt = li.get_text(separator=" ", strip=True)
        if len(txt) > 4:
            list_items.append(txt)

    body_text = main.get_text(separator="\n")

    combined = "\n\n".join(filter(None, [
        "\n".join(extra_parts),
        clean(body_text),
        ("Product list items:\n" + "\n".join(list_items)) if list_items else "",
    ]))
    return clean(combined)


# ─────────────────────────────────────────────────────────────────
# Ingest one source
# ─────────────────────────────────────────────────────────────────
def ingest_path(path: Path) -> list[dict]:
    ext = path.suffix.lower()
    log.info("Reading %s …", path)

    if ext == ".pdf":
        text = read_pdf(path)
    elif ext in {".txt", ".md", ".rst", ".html", ".htm"}:
        text = clean(read_txt(path))
    else:
        log.warning("Skipping unsupported file type: %s", path)
        return []

    if not text.strip():
        log.warning("No text extracted from %s", path)
        return []

    chunks = chunk_text(text, str(path))
    log.info("  → %d chunks from %s", len(chunks), path.name)
    return chunks


def ingest_url(url: str) -> list[dict]:
    text = read_url(url)
    if not text.strip():
        return []
    chunks = chunk_text(text, url)
    log.info("  → %d chunks from %s", len(chunks), url)
    return chunks


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build Maya knowledge base index")
    parser.add_argument("--url",  help="Extra URL to crawl and ingest")
    parser.add_argument("--file", help="Extra file to ingest")
    args = parser.parse_args()

    all_chunks: list[dict] = []

    # Config files that live in kb/ but must NOT be indexed as content
    KB_CONFIG_FILES = {"urls.txt"}

    # 1. Scan kb/ directory for documents
    if KB_DIR.exists():
        for f in sorted(KB_DIR.iterdir()):
            if f.name in KB_CONFIG_FILES:
                log.info("Skipping config file: %s", f.name)
                continue
            if f.suffix.lower() in {".pdf", ".txt", ".md", ".rst", ".html", ".htm"}:
                all_chunks.extend(ingest_path(f))
    else:
        KB_DIR.mkdir(exist_ok=True)
        log.info("Created kb/ — drop your PDFs, TXT, or MD files here.")

    # 2. Process URLs listed in kb/urls.txt (one per line)
    urls_file = KB_DIR / "urls.txt"
    if urls_file.exists():
        for line in urls_file.read_text().splitlines():
            url = line.strip()
            if url and not url.startswith("#"):
                all_chunks.extend(ingest_url(url))

    # 3. Extra CLI sources
    if args.url:
        all_chunks.extend(ingest_url(args.url))
    if args.file:
        all_chunks.extend(ingest_path(Path(args.file)))

    if not all_chunks:
        log.warning("No content found. Add files to kb/ or URLs to kb/urls.txt")
        sys.exit(0)

    # 4. Deduplicate by chunk id
    seen = set()
    unique = []
    for c in all_chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)

    # 5. Write index
    index = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_chunks": len(unique),
        "chunks": unique,
    }
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("✅ Wrote %d chunks → %s", len(unique), INDEX)
    log.info("   Restart (or let uvicorn --reload pick it up) to use the new index.")


if __name__ == "__main__":
    main()
