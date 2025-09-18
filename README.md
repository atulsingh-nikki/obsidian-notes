# Obsidian Notes – Bookshelf

This repository contains long-form research notes and two in-progress books:

- **Color Correction**
- **Object Detection**

To make the material easier to browse, the repository now includes a GitHub
Pages site that renders the Markdown chapters directly in the browser using
[Marked](https://marked.js.org/).

## Publishing to GitHub Pages

1. Push the latest changes to the `main` branch.
2. In the GitHub repository settings, open **Pages**.
3. Set the **Source** to `Deploy from a branch` → `main` → `/ (root)`.
4. Save the settings. GitHub will publish the site at `https://<username>.github.io/<repository>/`.

Because the site is built entirely from static HTML, CSS, and JavaScript, no
additional build step is required once Pages is enabled.

## Updating the navigation

The navigation sidebar is generated from `books.json`, which is produced by a
small helper script. Run the script whenever you add new chapters or rename
files:

```bash
python3 scripts/generate_books_index.py
```

Commit the updated `books.json` file alongside your content changes to keep the
site in sync with the repository.

## Local preview

You can preview the site locally before publishing by launching a simple HTTP
server from the repository root:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000> in your browser.
