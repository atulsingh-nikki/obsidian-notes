# Working in this repository

This is an Obsidian vault published as a **Jekyll site** (GitHub Pages, `cayman`
remote theme) at <https://atulsingh-nikki.github.io/obsidian-notes/>.

## Build / preview workflow

The site is data-driven. A generator must run **before** every Jekyll build:

```bash
python3 scripts/generate_site_data.py     # emit _data/*.json + assets/graph.json
bundle exec jekyll serve --baseurl /obsidian-notes
```

`scripts/generate_site_data.py` scans `_posts/` and `Research/` and writes:

- `_data/connections.json` — per-post `outlinks`, `backlinks`, `related` (shared-tag
  ranked), and `series` info. Keyed by post **slug** (the title part of the filename).
- `_data/series.json` — each series with its ordered parts.
- `_data/research.json` — research notes grouped by year + a coarse topic bucket.
- `assets/graph.json` — nodes + link edges for the `/map/` connections graph.

CI (`.github/workflows/pages.yml`) runs this generator step automatically before
`jekyll build`, then builds the Pagefind search index. If you add or re-link posts,
re-run the generator so backlinks/series/graph stay in sync.

Data is keyed by **slug**; final URLs are resolved at render time — in Liquid via
`site.posts | where: "slug", <slug>`, and in JS via the `window.__POST_URLS__` map
rendered on the `/map/` page. The generator never hardcodes permalinks.

## Layout / page structure

- `_layouts/default.html` — global chrome (top nav, footer, light/dark toggle). It
  **overrides** the remote theme's default layout.
- `_layouts/post.html` — reading layout (sticky TOC via `assets/reading.js`, progress
  bar, series/tag meta, and the chain region: series prev/next, related, backlinks).
- `_layouts/research-note.html` — applied to every file under `Research/` via a
  `_config.yml` `defaults` rule; turns each paper note into a styled page.
- Section pages: `books.html`, `research.html` (top-level files), `topics/`, `series/`,
  `map/`. Client JS: `assets/main.js` (books reader), `research.js`, `graph.js`.

## Gotchas (non-obvious — read before editing)

1. **macOS case-insensitive filesystem collides page dirs with vault dirs.** The vault
   content lives in `Books/` and `Research/` (capitalized). A Jekyll page whose output
   path differs only by case (e.g. a `research/` source dir, or `Research/Index.md`
   rendering to `/research/`) **collides locally** on macOS even though it works on the
   Linux CI runner. That is why the section pages are **top-level files** (`books.html`,
   `research.html`) with explicit `permalink:`, and why `Research/Index.md` is excluded.
   Do **not** create top-level `books/` or `research/` source directories.

2. **LaTeX pipes must be `\mid`.** A bare `|` in math breaks KaTeX and collides with
   Markdown table delimiters on GitHub Pages. See `.cursorrules` for the full rule. (A
   few older posts still use escaped `\|` inside Liquid `{{ ... }}` image tags, which
   Jekyll logs as harmless Liquid warnings at build time.)

3. **Junk stays out of the build via `_config.yml` `exclude`.** `Current Status.md`,
   `Deep Research Prompt.md`, `README.md`, `Untitled.base`, templates, `scripts/`, etc.
   are excluded so they don't publish as orphaned URLs. Research note `.md` files are
   **not** excluded — they must remain for their pages to render.

4. **Research notes all carry front matter**, so Jekyll renders them to `.html`. Link to
   the built `.html` (not the raw `.md`); the `research-note` layout provides chrome.

5. **iCloud sync** can leave `Foo 2.md` conflict copies and empty `Foo 2/` folders. These
   are duplicates — remove them; don't treat them as content.
