from pathlib import Path
import hashlib
import json
import re


RAW_DIR = Path("data/raw/python")
PROCESSED_DIR = Path("data/processed")

OUTPUT_FILE = PROCESSED_DIR / "python_train.txt"
STATS_FILE = PROCESSED_DIR / "python_stats.json"


def normalize_text(text: str) -> str:
    """
    Normalize scraped documentation text.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing whitespace
    lines = [
        line.rstrip()
        for line in text.splitlines()
    ]

    text = "\n".join(lines)

    # Remove excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


def is_valid_document(text: str) -> bool:
    """
    Basic quality checks for documentation.
    """

    if not text:
        return False

    # Ignore extremely small files
    if len(text) < 100:
        return False

    return True


def content_hash(text: str) -> str:
    """
    Used for exact duplicate detection.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def process():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    seen_hashes = set()

    documents = []

    stats = {
        "files_found": 0,
        "files_kept": 0,
        "duplicates": 0,
        "invalid": 0,
        "characters": 0,
        "estimated_tokens": 0,
    }

    # Find all text files
    files = list(RAW_DIR.rglob("*.txt"))

    stats["files_found"] = len(files)

    print(f"Found {len(files)} text files")

    for path in files:

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            continue

        # Normalize
        text = normalize_text(text)

        # Quality check
        if not is_valid_document(text):
            stats["invalid"] += 1
            continue

        # Exact duplicate check
        file_hash = content_hash(text)

        if file_hash in seen_hashes:
            stats["duplicates"] += 1
            continue

        seen_hashes.add(file_hash)

        documents.append(text)

        stats["files_kept"] += 1
        stats["characters"] += len(text)

    # ------------------------------------------------
    # Write processed corpus
    # ------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        for document in documents:

            # Document boundary
            f.write(document)

            f.write("\n\n")
            f.write("# === DOCUMENT END ===")
            f.write("\n\n")

    # Rough estimate only.
    # Actual count will come from the tokenizer later.
    stats["estimated_tokens"] = (
        stats["characters"] // 4
    )

    # Save statistics
    STATS_FILE.write_text(
        json.dumps(
            stats,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nProcessing complete")
    print("-------------------")
    print(f"Files found:       {stats['files_found']:,}")
    print(f"Files kept:        {stats['files_kept']:,}")
    print(f"Duplicates:        {stats['duplicates']:,}")
    print(f"Invalid:           {stats['invalid']:,}")
    print(f"Characters:        {stats['characters']:,}")
    print(
        f"Estimated tokens:  "
        f"{stats['estimated_tokens']:,}"
    )

    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    process()