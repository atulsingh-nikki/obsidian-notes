---
layout: post
title: "How the Learning Collection Gets Built Each Week"
description: "A peek at the scripts and structure that keep the learning projects organized."
tags: [process, automation]
---

One of the benefits of keeping research inside Obsidian is that the vault can double as a content pipeline. Folders map to learning projects, daily notes track reading sessions, and scripts update the public learning collection automatically.

Every Sunday a GitHub Action regenerates `books.json` so that new chapters appear in the sidebar without any manual curation. The script walks the repository looking for folders that start with `Book_`, pulls their metadata, and pushes the results back into the site. The front-end JavaScript then renders the navigation tree and loads Markdown chapters on demand.

This automation means the GitHub Pages site is always current, even when only a handful of paragraphs changed during the week. Future posts will explore how research notes feed into longer essays and what additional tooling could make that handoff smoother.
