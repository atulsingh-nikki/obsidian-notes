#!/usr/bin/env python3
"""Generate connection / series / research data + a graph for the site.

Scans _posts and Research once and emits Jekyll data files consumed by the
layouts and section pages, plus assets/graph.json for the visual map.

Outputs (all keyed by post *slug* — the title portion of the filename — so the
layouts/JS can resolve slug -> url/title without replicating Jekyll permalinks):
  _data/connections.json   per-post outlinks, backlinks, related, series
  _data/series.json        ordered series -> parts
  _data/research.json      research notes grouped by year + topic buckets
  assets/graph.json        nodes + link edges for /map/

Pure stdlib (no PyYAML) so it runs unchanged in CI.
"""

import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSTS_DIR = os.path.join(ROOT, "_posts")
RESEARCH_DIR = os.path.join(ROOT, "Research")
DATA_DIR = os.path.join(ROOT, "_data")
ASSETS_DIR = os.path.join(ROOT, "assets")
SERIES_DEF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "series.json")

POST_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)\.md$")
LINK_RE = re.compile(r"\{%\s*link\s+_posts/([^\s%]+)\.md\s*%\}")
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Skip non-note files that live under Research/ or _posts/.
SKIP_BASENAMES = {
    "deep research prompt.md",
    "current status.md",
    "researchtempalte.md",
    "index.md",
    "readme.md",
}

# Coarse topic buckets for Research notes, matched against the title (first hit wins).
RESEARCH_TOPICS = [
    ("Detection", ["detection", "detector", "r-cnn", "rcnn", "yolo", "ssd", "proposal", "overfeat", "edgebox"]),
    ("Segmentation", ["segment", "mask", "matting", "panoptic", "semantic"]),
    ("Tracking", ["tracking", "track", "mot", "sort", "reid", "re-id"]),
    ("3D / SLAM / NeRF", ["slam", "nerf", "radiance", "3d", "depth", "stereo", "structure from motion", "sfm", "pose"]),
    ("Color / Imaging", ["color", "retinex", "constancy", "hdr", "tone", "white balance", "gamut", "demosaic"]),
    ("Generative", ["gan", "diffusion", "generative", "vae", "autoencoder", "flow"]),
    ("Backbones / Representation", ["resnet", "vgg", "convnet", "convnext", "inception", "googlenet", "transformer", "vit", "embedding", "representation", "self-supervised", "dino", "clip"]),
    ("Estimation / Filtering", ["kalman", "filter", "bayes", "estimation", "particle"]),
    ("Recognition / Classification", ["recognition", "classification", "ocr", "label", "scene"]),
]


def parse_front_matter(text):
    """Return (title, tags[], description) from a post's front matter."""
    m = FRONT_MATTER_RE.match(text)
    title, tags, desc = None, [], None
    if not m:
        return title, tags, desc
    block = m.group(1)
    for line in block.splitlines():
        line = line.rstrip()
        if line.startswith("title:"):
            title = line[len("title:"):].strip().strip('"').strip("'")
        elif line.startswith("description:"):
            desc = line[len("description:"):].strip().strip('"').strip("'")
        elif line.startswith("tags:"):
            raw = line[len("tags:"):].strip()
            raw = raw.strip("[]")
            tags = [t.strip().strip('"').strip("'") for t in raw.split(",") if t.strip()]
    return title, tags, desc


def load_posts():
    posts = {}
    for fn in sorted(os.listdir(POSTS_DIR)):
        m = POST_NAME_RE.match(fn)
        if not m:
            continue
        date, slug = m.group(1), m.group(2)
        path = os.path.join(POSTS_DIR, fn)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        title, tags, desc = parse_front_matter(text)
        outlinks = []
        for ref in LINK_RE.findall(text):
            rm = POST_NAME_RE.match(ref + ".md")
            if rm and rm.group(2) not in outlinks:
                outlinks.append(rm.group(2))
        posts[slug] = {
            "slug": slug,
            "date": date,
            "title": title or slug.replace("-", " ").title(),
            "tags": [t for t in tags if t != "series"],
            "outlinks": outlinks,
            "series": None,  # filled in from series.json by build_series()
        }
    return posts


def build_series(posts):
    """Build series from the authored scripts/series.json (ordered slug lists),
    and stamp each member post with its series/part/prev/next."""
    with open(SERIES_DEF, "r", encoding="utf-8") as f:
        definitions = json.load(f)

    series_list = []
    for d in definitions:
        parts = [s for s in d["slugs"] if s in posts]  # keep only existing posts, in order
        if not parts:
            continue
        total = len(parts)
        entry = {
            "name": d["name"],
            "description": d.get("description", ""),
            "parts": [
                {"slug": s, "part": i + 1, "title": posts[s]["title"]}
                for i, s in enumerate(parts)
            ],
        }
        series_list.append(entry)
        for i, s in enumerate(parts):
            posts[s]["series"] = {
                "name": d["name"],
                "part": i + 1,
                "total": total,
                "prev": parts[i - 1] if i > 0 else None,
                "next": parts[i + 1] if i < total - 1 else None,
            }
    return series_list


def build_connections(posts):
    backlinks = defaultdict(list)
    for slug, p in posts.items():
        for target in p["outlinks"]:
            if target in posts and slug not in backlinks[target]:
                backlinks[target].append(slug)

    connections = {}
    for slug, p in posts.items():
        tagset = set(p["tags"])
        scored = []
        if tagset:
            for other_slug, other in posts.items():
                if other_slug == slug:
                    continue
                shared = tagset & set(other["tags"])
                if shared:
                    scored.append((len(shared), other["date"], other_slug))
        scored.sort(key=lambda x: (-x[0], x[1] < p["date"], x[2]))
        related = [s for _, _, s in scored[:6]]
        connections[slug] = {
            "title": p["title"],
            "outlinks": [t for t in p["outlinks"] if t in posts],
            "backlinks": backlinks.get(slug, []),
            "related": related,
            "series": p["series"],
        }
    return connections


def research_topic(title):
    low = title.lower()
    for label, keys in RESEARCH_TOPICS:
        if any(k in low for k in keys):
            return label
    return "Other"


def clean_research_title(fn):
    name = fn[:-3] if fn.endswith(".md") else fn
    # Strip a trailing " (…year…)" or " (2021-2022)" style suffix.
    name = re.sub(r"\s*\([^()]*\b(19|20)\d{2}[^()]*\)\s*$", "", name).strip()
    return name or fn


def load_research():
    by_year = defaultdict(list)
    topics = set()
    count = 0
    if not os.path.isdir(RESEARCH_DIR):
        return {"by_year": {}, "topics": [], "count": 0}
    for dirpath, _dirs, files in os.walk(RESEARCH_DIR):
        year_dir = os.path.basename(dirpath)
        if not re.fullmatch(r"(19|20)\d{2}", year_dir):
            continue
        for fn in sorted(files):
            if not fn.endswith(".md") or fn.lower() in SKIP_BASENAMES:
                continue
            title = clean_research_title(fn)
            topic = research_topic(title)
            topics.add(topic)
            rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
            by_year[year_dir].append({"title": title, "path": rel, "topic": topic})
            count += 1
    ordered = {y: by_year[y] for y in sorted(by_year, reverse=True)}
    return {"by_year": ordered, "topics": sorted(topics), "count": count}


def build_graph(posts):
    nodes = []
    for slug, p in posts.items():
        group = p["series"]["name"] if p["series"] else (p["tags"][0] if p["tags"] else "misc")
        nodes.append({
            "id": slug,
            "title": p["title"],
            "tags": p["tags"],
            "group": group,
        })
    seen = set()
    links = []
    for slug, p in posts.items():
        for target in p["outlinks"]:
            if target in posts:
                key = tuple(sorted((slug, target)))
                if key not in seen:
                    seen.add(key)
                    links.append({"source": slug, "target": target})
    degree = defaultdict(int)
    for l in links:
        degree[l["source"]] += 1
        degree[l["target"]] += 1
    for n in nodes:
        n["degree"] = degree.get(n["id"], 0)
    return {"nodes": nodes, "links": links}


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    posts = load_posts()
    series = build_series(posts)  # mutates posts[*]["series"] with prev/next
    connections = build_connections(posts)
    research = load_research()
    graph = build_graph(posts)

    write_json(os.path.join(DATA_DIR, "connections.json"), connections)
    write_json(os.path.join(DATA_DIR, "series.json"), series)
    write_json(os.path.join(DATA_DIR, "research.json"), research)
    write_json(os.path.join(ASSETS_DIR, "graph.json"), graph)

    total_links = sum(len(c["outlinks"]) for c in connections.values())
    print(f"posts: {len(posts)}  inline-link edges: {total_links}")
    print(f"series: {len(series)} ({', '.join(s['name'] + ' x' + str(len(s['parts'])) for s in series)})")
    print(f"research notes: {research['count']} across {len(research['by_year'])} years")
    print(f"graph: {len(graph['nodes'])} nodes, {len(graph['links'])} links")


if __name__ == "__main__":
    main()
