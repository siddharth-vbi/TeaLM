from pathlib import Path
from urllib.parse import urlparse, urljoin
import json
import re

import requests
import trafilatura
from bs4 import BeautifulSoup


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data/raw/python2")
METADATA_FILE = DATA_DIR / "metadata.jsonl"

SOURCES = {
    "python": [
        # Add permitted documentation URLs here
        "https://www.w3schools.com/python/",
    ],

    "javascript": [
        # Add permitted documentation URLs here
    ],

    "typescript": [
        # Add permitted documentation URLs here
    ],

    "react": [
        # Add permitted documentation URLs here
    ],

    "git": [
        # Add permitted documentation URLs here
    ],

    "linux": [
        # Add permitted documentation URLs here
    ],
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def safe_filename(url: str) -> str:
    """
    Convert URL into a safe filename.
    """

    parsed = urlparse(url)

    name = parsed.path.strip("/")

    if not name:
        name = parsed.netloc

    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)

    return name[:100] + ".txt"


def download_page(url: str) -> str | None:
    """
    Download HTML from URL.
    """

    headers = {
        "User-Agent": (
            "CodeLM-Dataset-Collector/1.0 "
            "(educational research project)"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()

        return response.text

    except requests.RequestException as e:
        print(f"[ERROR] Download failed: {url}")
        print(f"        {e}")

        return None


def extract_text(html: str) -> str | None:
    """
    Extract main readable content from HTML.
    """

    text = trafilatura.extract(
        html,
        include_links=False,
        include_images=False,
        include_tables=True,
    )

    if not text:
        return None

    return text.strip()


def extract_pagination_links(html: str, base_url: str) -> list[str]:
    """
    Extract pagination/navigation links from HTML.
    Returns a list of absolute URLs for next, previous, and numbered pages.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = []

    # Common pagination patterns
    # 1. Next/Previous buttons with rel attributes
    for link in soup.find_all("a", rel=["next", "prev"]):
        href = link.get("href")
        if href:
            links.append(urljoin(base_url, href))

    # 2. Links with pagination classes
    pagination_classes = [
        "pagination", "pager", "page-numbers", "nav-links",
        "next-page", "prev-page", "page-link", "pagination-link"
    ]
    for class_name in pagination_classes:
        for link in soup.find_all("a", class_=re.compile(class_name, re.I)):
            href = link.get("href")
            if href:
                links.append(urljoin(base_url, href))

    # 3. Next/Previous buttons by text content
    for link in soup.find_all("a"):
        text = link.get_text(strip=True).lower()
        if text in ["next", "previous", "prev", "next page", "previous page", "older", "newer"]:
            href = link.get("href")
            if href:
                links.append(urljoin(base_url, href))

    # 4. Numbered page links (often in a nav or pagination container)
    nav_containers = soup.find_all(["nav", "div", "ul"], class_=re.compile(r"pagination|pager|pages", re.I))
    for container in nav_containers:
        for link in container.find_all("a"):
            href = link.get("href")
            if href:
                links.append(urljoin(base_url, href))

    # Deduplicate while preserving order
    seen = set()
    unique_links = []
    for link in links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    return unique_links


def collect_paginated(
    start_url: str,
    category: str,
    visited: set[str] | None = None,
) -> list[tuple[str, str, Path]]:
    """
    Collect all pages from a paginated series starting at start_url.
    Returns list of (url, filename, path) tuples.
    """
    if visited is None:
        visited = set()

    results = []
    current_url = start_url

    while current_url and current_url not in visited:
        visited.add(current_url)

        print(f"  Downloading: {current_url}")
        html = download_page(current_url)

        if html is None:
            break

        text = extract_text(html)
        if not text:
            print(f"  [SKIP] Could not extract text from {current_url}")
            break

        filename = safe_filename(current_url)
        path = save_text(category, filename, text)
        save_metadata(category, current_url, path, text)

        results.append((current_url, filename, path))

        # Find pagination links
        pagination_links = extract_pagination_links(html, current_url)

        # Determine next URL - prefer "next" rel, then text-based "next", then first unvisited
        next_url = None
        for link in pagination_links:
            if link not in visited:
                next_url = link
                break

        if next_url == current_url:
            break  # Avoid self-loops

        current_url = next_url

    return results


def save_text(
    category: str,
    filename: str,
    text: str,
) -> Path:
    """
    Save extracted text.
    """

    category_dir = DATA_DIR / category
    category_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = category_dir / filename

    path.write_text(
        text,
        encoding="utf-8",
    )

    return path


def save_metadata(
    category: str,
    url: str,
    path: Path,
    text: str,
):
    """
    Store information about the collected document.
    """

    record = {
        "category": category,
        "url": url,
        "file": str(path),
        "characters": len(text),
        "estimated_tokens": len(text) // 4,
    }

    with METADATA_FILE.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )


# --------------------------------------------------
# Main collector
# --------------------------------------------------

def collect():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_chars = 0

    visited_global: set[str] = set()

    for category, urls in SOURCES.items():

        print()
        print("=" * 60)
        print(f"Category: {category}")
        print("=" * 60)

        for url in urls:

            print(f"\nCollecting (with pagination): {url}")

            # Collect all pages in the paginated series
            results = collect_paginated(url, category, visited_global)

            for page_url, filename, path in results:
                # Re-read to get stats (or pass them from collect_paginated)
                text = path.read_text(encoding="utf-8")
                chars = len(text)
                tokens = chars // 4
                total_chars += chars

                print(f"[SAVED] {path}")
                print(f"        Characters: {chars:,}")
                print(f"        Est. tokens: {tokens:,}")

            if not results:
                print(f"[SKIP] No content collected from {url}")

    print()
    print("=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)

    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: {total_chars // 4:,}")


if __name__ == "__main__":
    collect()