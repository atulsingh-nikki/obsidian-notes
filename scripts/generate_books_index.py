#!/usr/bin/env python3
"""Generate navigation data for the book collection.

The script scans every directory that starts with ``Book_`` and produces a JSON
file (``books.json``) that contains metadata for the GitHub Pages site.  Each
book includes the Markdown files that belong to it so that the front-end can
load and render them on demand.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "books.json"
BOOK_PREFIX = "Book_"
IGNORED_DIRECTORIES = {"images", "img", "assets", "static"}


def natural_key(text: str) -> List[object]:
    """Split text into digit and non-digit chunks for natural sorting."""
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", text)]


def natural_key_path(path: Path, base: Path) -> List[object]:
    """Generate a natural sort key for a path relative to ``base``."""
    parts: List[object] = []
    for piece in path.relative_to(base).parts:
        parts.extend(natural_key(piece))
    return parts


def format_book_title(folder_name: str) -> str:
    """Return a human-friendly book title from the folder name."""
    name = folder_name[len(BOOK_PREFIX) :] if folder_name.startswith(BOOK_PREFIX) else folder_name
    name = name.replace("_", " ")
    return name.strip()


def format_section_title(name: str) -> str:
    """Create a display title for a section directory."""
    readable = name.replace("_", " ").strip()
    match = re.match(r"^(Part\s+[\wIVXLC]+)(?:\s+)(.*)$", readable)
    if match:
        prefix, remainder = match.groups()
        return f"{prefix}: {remainder}" if remainder else prefix
    return readable


def insert_spaces_from_camel_case(text: str) -> str:
    """Insert spaces between camelCase or PascalCase boundaries."""
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r" \1", text)


def format_item_title(source: str) -> str:
    """Convert a file stem into a readable title."""
    readable = source.replace("_", " ").strip()
    if "_" not in source and re.search(r"[a-z][A-Z]", source):
        readable = insert_spaces_from_camel_case(source)
    match = re.match(r"^(Chapter\s+[\wIVXLC]+)(?:\s+)(.*)$", readable)
    if match:
        prefix, remainder = match.groups()
        if remainder:
            return f"{prefix} – {remainder}"
        return prefix
    normalised = readable.replace(" ", "").lower()
    if normalised == "tableofcontents":
        return "Table of Contents"
    return readable


def discover_markdown_files(directory: Path) -> List[Path]:
    """Return all Markdown files under ``directory`` ignoring auxiliary folders."""
    markdown_files = []
    for path in directory.rglob("*.md"):
        if any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue
        markdown_files.append(path)
    return markdown_files


def iter_book_directories() -> Iterable[Path]:
    """Yield directories that contain book content.

    Historically the repository stored ``Book_*`` folders at the project root,
    but they have since been grouped inside a ``Books`` directory.  To remain
    backwards compatible and support both layouts we look in each location.
    Duplicate directories are ignored in case a symlink is present.
    """

    search_roots = [ROOT]
    books_folder = ROOT / "Books"
    if books_folder.is_dir():
        search_roots.append(books_folder)

    seen: set[Path] = set()
    for base in search_roots:
        for candidate in base.iterdir():
            if not candidate.is_dir() or not candidate.name.startswith(BOOK_PREFIX):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield candidate


def build_book(book_dir: Path) -> dict:
    """Build the data structure describing a single book."""
    sections = []

    overview_files = sorted(book_dir.glob("*.md"), key=lambda p: natural_key(p.stem))
    if overview_files:
        sections.append(
            {
                "title": "Overview",
                "items": [make_item(file_path) for file_path in overview_files],
            }
        )

    part_dirs = [
        child
        for child in book_dir.iterdir()
        if child.is_dir() and not child.name.startswith(".") and child.name not in IGNORED_DIRECTORIES
    ]

    for part_dir in sorted(part_dirs, key=lambda p: natural_key(p.name)):
        markdown_files = discover_markdown_files(part_dir)
        if not markdown_files:
            continue
        sections.append(
            {
                "title": format_section_title(part_dir.name),
                "items": [
                    make_item(md_file, display_root=part_dir)
                    for md_file in sorted(markdown_files, key=lambda p: natural_key_path(p, part_dir))
                ],
            }
        )

    return {
        "title": format_book_title(book_dir.name),
        "path": book_dir.relative_to(ROOT).as_posix(),
        "sections": sections,
    }


def make_item(path: Path, display_root: Path | None = None) -> dict:
    """Create the JSON representation for a markdown file."""
    stem = path.stem
    if stem.lower() == "readme":
        stem = display_root.name if display_root is not None else path.parent.name

    parent_name = path.parent.name
    if display_root is not None and path.parent != display_root:
        stem = parent_name
    elif re.match(r"Chapter[_-]", stem, re.IGNORECASE) and parent_name.lower().startswith(stem.lower()):
        stem = parent_name

    return {
        "title": format_item_title(stem),
        "path": path.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    book_dirs = sorted(iter_book_directories(), key=lambda p: natural_key(p.name))
    books = [build_book(book_dir) for book_dir in book_dirs]

    OUTPUT.write_text(json.dumps(books, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(books)} books.")


if __name__ == "__main__":
    main()
