# Atul Singh Notes

This repository contains long-form research notes, two in-progress learning projects, and a blog for shorter updates.

👉 View the published site on GitHub Pages: <https://atulsingh-nikki.github.io/obsidian-notes/>

🗂️ Need a birds-eye view? Consult the [Publishing Cadence Summary](https://atulsingh-nikki.github.io/obsidian-notes/2025/03/10/publishing-cadence-summary.html) for monthly, quarterly, semiannual, and annual rollups.

The collection currently features:

- **Color Correction**
- **Object Detection**

The accompanying blog highlights release notes, process updates, and essays that add context to the evolving library of notes.

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
learning collection in sync with the repository.

## Writing blog posts

1. Create a new Markdown file inside the `_posts/` directory using the
   `YYYY-MM-DD-title.md` naming convention.
2. Add front matter similar to the example below. The `layout: post` line
   applies the custom article layout, while the optional `tags` field powers the
   blog index metadata.

   ```yaml
   ---
   layout: post
   title: "A short update"
   description: "What changed inside the vault this week."
   tags: [updates]
   ---
   ```

3. Write the body of the post using standard Markdown.
4. Commit the new file and push to `main`. GitHub Pages will automatically
   publish the post at `/blog/`.

## Local preview

You can preview the site locally before publishing by launching a simple HTTP
server from the repository root:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000> in your browser.
