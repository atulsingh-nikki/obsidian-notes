const state = {
  books: [],
  activePath: null,
  selectedBook: null,
  viewMode: 'overview', // 'overview' or 'chapter'
};

const siteConfig = window.__SITE_CONFIG__ || {};
const booksIndexUrl = siteConfig.booksIndexUrl || "books.json";

const navigationEl = document.getElementById("book-navigation");
const contentEl = document.getElementById("book-content");
const mainEl = document.getElementById("main");
const booksColumnEl = document.querySelector(".home-page__column--books");
const sidebarIntroEl = document.querySelector(".bookshelf-sidebar__intro");
const navRegistry = new Map();

// Toggle the books column between the shelf (card grid) and the reader (sidebar + content).
function setBooksMode(mode) {
  if (booksColumnEl) {
    booksColumnEl.classList.toggle("is-shelf", mode === "shelf");
    booksColumnEl.classList.toggle("is-reading", mode === "reading");
    // Always reveal the chapter nav when (re)entering a book.
    if (mode !== "reading") {
      booksColumnEl.classList.remove("sidebar-hidden");
    }
  }
  if (sidebarIntroEl) {
    sidebarIntroEl.hidden = mode !== "reading";
  }
  if (mainEl) {
    mainEl.style.display = mode === "reading" ? "" : "none";
  }
  if (mode === "reading") {
    ensureSidebarToggle();
    updateSidebarToggle();
  }
}

// Add the "Contents" show/hide control once, above the chapter content.
function ensureSidebarToggle() {
  if (!mainEl || document.getElementById("sidebar-toggle")) return;
  const button = document.createElement("button");
  button.id = "sidebar-toggle";
  button.type = "button";
  button.className = "bookshelf-toggle";
  button.setAttribute("aria-controls", "sidebar");
  button.addEventListener("click", () => {
    if (!booksColumnEl) return;
    booksColumnEl.classList.toggle("sidebar-hidden");
    updateSidebarToggle();
  });
  mainEl.insertBefore(button, mainEl.firstChild);
}

function updateSidebarToggle() {
  const button = document.getElementById("sidebar-toggle");
  if (!button || !booksColumnEl) return;
  const hidden = booksColumnEl.classList.contains("sidebar-hidden");
  button.setAttribute("aria-expanded", String(!hidden));
  button.textContent = hidden ? "☰ Show contents" : "✕ Hide contents";
}

if (window.marked) {
  window.marked.setOptions({
    breaks: false,
    gfm: true,
    mangle: false,
    headerIds: true,
  });
}

async function init() {
  if (!navigationEl || !contentEl || !mainEl) {
    return;
  }
  try {
    const books = await loadBooks();
    state.books = books;
    if (books.length === 0) {
      renderNavigationPlaceholder("No learning collection directories were found. Add folders that start with 'Book_'.");
      renderContentMessage("No content to display yet.");
      return;
    }

    const initialPath = decodeHash(location.hash);
    if (initialPath) {
      // If there's a hash, show the chapter view
      buildNavigation(books);
      await displayChapter(initialPath);
    } else {
      // Show books overview by default
      showBooksOverview();
    }
  } catch (error) {
    console.error(error);
    renderNavigationPlaceholder("Unable to load learning collection metadata. Ensure books.json exists.");
    renderError(String(error));
  }
}

async function loadBooks() {
  const response = await fetch(resolveUrl(booksIndexUrl), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`books.json could not be loaded (HTTP ${response.status})`);
  }
  return response.json();
}

function showBooksOverview() {
  state.viewMode = 'overview';
  state.selectedBook = null;
  state.activePath = null;
  history.replaceState(null, "", window.location.pathname);

  setBooksMode('shelf');
  renderBooksOverview();
  showBlogSection();
}

function renderBooksOverview() {
  navigationEl.innerHTML = "";
  
  const overviewContainer = document.createElement("div");
  overviewContainer.className = "books-overview";
  
  const header = document.createElement("div");
  header.className = "books-overview__header";
  header.innerHTML = `
    <h2>Available Books</h2>
    <p>Select a book to explore its chapters and content.</p>
  `;
  overviewContainer.appendChild(header);
  
  const grid = document.createElement("div");
  grid.className = "books-overview__grid";
  
  const bookDescriptions = {
    'Color Correction': 'A comprehensive guide to color science, digital imaging pipeline, and professional color grading workflows.',
    'Object Detection': 'Explore computer vision fundamentals, detection architectures, segmentation techniques, and tracking algorithms.',
    'Deep Learning Architectures': 'Learn about modern neural network architectures and their applications in various domains.',
    'Training Strategies': 'Master optimization techniques, regularization, transfer learning, and advanced training strategies for computer vision models.'
  };
  
  const bookIcons = ['🎨', '👁️', '🧠', '🎯', '🔬', '📊', '📚'];
  
  state.books.forEach((book, index) => {
    const card = document.createElement("div");
    card.className = "book-card";
    card.setAttribute("role", "button");
    card.setAttribute("tabindex", "0");
    
    const totalChapters = book.sections.reduce((sum, section) => sum + section.items.length, 0);
    const sectionCount = book.sections.length;
    const description = bookDescriptions[book.title] || `Explore ${book.title.toLowerCase()} with detailed chapters and sections.`;
    
    card.innerHTML = `
      <div class="book-card__header">
        <div class="book-card__icon">${bookIcons[index % bookIcons.length]}</div>
        <h3 class="book-card__title">${book.title}</h3>
      </div>
      <p class="book-card__description">${description}</p>
      <div class="book-card__meta">
        <span class="book-card__meta-item">📖 ${totalChapters} chapters</span>
        <span class="book-card__meta-item">📑 ${sectionCount} sections</span>
      </div>
      <p class="book-card__arrow">Explore book →</p>
    `;
    
    card.addEventListener("click", () => selectBook(book));
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectBook(book);
      }
    });
    
    grid.appendChild(card);
  });
  
  overviewContainer.appendChild(grid);
  navigationEl.appendChild(overviewContainer);
}

function selectBook(book) {
  state.viewMode = 'chapter';
  state.selectedBook = book;
  buildNavigation(state.books);
  hideBlogSection();
  setBooksMode('reading');

  // Load the first chapter of the selected book
  const firstPath = firstChapterPathForBook(book);
  if (firstPath) {
    navigateToChapter(firstPath);
  }
}

function firstChapterPathForBook(book) {
  for (const section of book.sections) {
    if (section.items && section.items.length > 0) {
      return section.items[0].path;
    }
  }
  return "";
}

function renderBreadcrumb(bookTitle) {
  const breadcrumb = document.createElement("div");
  breadcrumb.className = "bookshelf-breadcrumb";
  
  const homeLink = document.createElement("a");
  homeLink.href = "#";
  homeLink.textContent = "All Books";
  homeLink.addEventListener("click", (e) => {
    e.preventDefault();
    showBooksOverview();
  });
  
  const separator = document.createElement("span");
  separator.className = "bookshelf-breadcrumb__separator";
  separator.textContent = "/";
  
  const current = document.createElement("span");
  current.className = "bookshelf-breadcrumb__current";
  current.textContent = bookTitle;
  
  breadcrumb.appendChild(homeLink);
  breadcrumb.appendChild(separator);
  breadcrumb.appendChild(current);
  
  return breadcrumb;
}

function buildNavigation(books) {
  navigationEl.innerHTML = "";
  navRegistry.clear();

  // Add breadcrumb if we're viewing a specific book
  if (state.selectedBook) {
    navigationEl.appendChild(renderBreadcrumb(state.selectedBook.title));
  }

  const booksToShow = state.selectedBook ? [state.selectedBook] : books;

  booksToShow.forEach((book, bookIndex) => {
    const bookContainer = document.createElement("div");
    bookContainer.className = "bookshelf-book";

    // Only show book title if displaying multiple books
    if (!state.selectedBook) {
      const bookHeader = document.createElement("h3");
      bookHeader.textContent = book.title;
      bookContainer.appendChild(bookHeader);
    }

    book.sections.forEach((section, sectionIndex) => {
      const details = document.createElement("details");
      details.className = "bookshelf-section";
      if (bookIndex === 0 && sectionIndex === 0) {
        details.open = true;
      }

      const summary = document.createElement("summary");
      summary.textContent = section.title;
      details.appendChild(summary);

      const list = document.createElement("ul");
      section.items.forEach((item) => {
        const listItem = document.createElement("li");
        const link = document.createElement("a");
        link.href = `#${encodeURIComponent(item.path)}`;
        link.textContent = item.title;
        link.dataset.path = item.path;
        link.addEventListener("click", (event) => {
          event.preventDefault();
          navigateToChapter(item.path);
        });
        listItem.appendChild(link);
        list.appendChild(listItem);

        navRegistry.set(item.path, { link, details });
      });

      details.appendChild(list);
      bookContainer.appendChild(details);
    });

    navigationEl.appendChild(bookContainer);
  });
}

function navigateToChapter(path) {
  if (!path) return;
  updateHash(path, { replace: false });
}

function updateHash(path, { replace }) {
  const encoded = `#${encodeURIComponent(path)}`;
  if (replace) {
    history.replaceState(null, "", encoded);
  } else {
    if (location.hash === encoded) {
      // Trigger hashchange manually because browsers do not emit it when the value is unchanged.
      handleHashChange();
    } else {
      location.hash = encoded;
    }
  }
}

function decodeHash(hash) {
  if (!hash || hash.length <= 1) return "";
  try {
    return decodeURIComponent(hash.slice(1));
  } catch (error) {
    console.warn("Unable to decode hash", hash, error);
    return "";
  }
}

async function displayChapter(path) {
  if (!path) return;
  
  // Find which book this chapter belongs to
  if (!state.selectedBook) {
    for (const book of state.books) {
      for (const section of book.sections) {
        for (const item of section.items) {
          if (item.path === path) {
            state.selectedBook = book;
            state.viewMode = 'chapter';
            buildNavigation(state.books);
            break;
          }
        }
        if (state.selectedBook) break;
      }
      if (state.selectedBook) break;
    }
  }
  
  // Hide blog section and switch to the reader when viewing a chapter
  hideBlogSection();
  setBooksMode('reading');

  const registryEntry = navRegistry.get(path);
  if (!registryEntry) {
    renderError("The requested chapter is not part of the current index.");
    return;
  }

  setActiveLink(path, registryEntry);
  await loadChapter(path);
}

function setActiveLink(path, entry) {
  if (state.activePath && navRegistry.has(state.activePath)) {
    const previous = navRegistry.get(state.activePath);
    previous.link.classList.remove("active");
  }

  state.activePath = path;
  entry.link.classList.add("active");
  entry.details.open = true;
  entry.link.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

async function loadChapter(path) {
  renderLoading();
  try {
    const response = await fetch(resolveUrl(path));
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    let markdown = await response.text();
    markdown = stripFrontMatter(markdown);
    // Protect LaTeX from the Markdown parser (Marked treats "_" as emphasis and
    // mangles subscripts before MathJax ever sees the math). Extract math spans,
    // let Marked run, then restore the raw TeX for MathJax to typeset.
    const math = [];
    const protectedMd = protectMath(markdown, math);
    let html = window.marked ? window.marked.parse(protectedMd) : protectedMd;
    html = restoreMath(html, math);
    contentEl.innerHTML = html;
    wrapTables(contentEl);
    markImageLoadErrors(contentEl);
    await typesetDynamicContent(contentEl);
    try {
      mainEl.focus({ preventScroll: true });
    } catch (error) {
      mainEl.focus();
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (error) {
    renderError(`Unable to load the selected chapter. ${error.message}`);
  }
}

// Wrap tables so wide tables can scroll horizontally instead of overflowing the layout.
function wrapTables(container) {
  container.querySelectorAll("table").forEach((table) => {
    if (table.parentElement && table.parentElement.classList.contains("bookshelf-table-wrap")) {
      return;
    }
    const wrapper = document.createElement("div");
    wrapper.className = "bookshelf-table-wrap";
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  });
}

// Flag images that fail to load so they degrade gracefully instead of showing a broken-image icon.
// Also rewrite repo-root-relative image sources (e.g. "Books/…/fig.png") through the
// site baseurl, so figures resolve correctly on the /books/ reader page and on the
// case-sensitive production host — not just on a case-insensitive local filesystem.
function markImageLoadErrors(container) {
  container.querySelectorAll("img").forEach((img) => {
    const raw = img.getAttribute("src") || "";
    if (raw && !/^(?:[a-z]+:)?\/\//i.test(raw) && !raw.startsWith("data:")) {
      img.src = resolveUrl(raw);
    }
    img.loading = "lazy";
    img.addEventListener("error", () => {
      img.classList.add("bookshelf-image--broken");
    }, { once: true });
  });
}

// Re-run MathJax and Mermaid on dynamically inserted content; neither library watches the DOM automatically.
async function typesetDynamicContent(container) {
  if (window.MathJax && window.MathJax.typesetPromise) {
    try {
      await window.MathJax.typesetPromise([container]);
    } catch (error) {
      console.warn("MathJax typesetting failed", error);
    }
  }

  const mermaidBlocks = container.querySelectorAll("pre code.language-mermaid, pre code.lang-mermaid");
  if (mermaidBlocks.length > 0 && window.mermaid) {
    mermaidBlocks.forEach((code) => {
      const pre = code.parentElement;
      const diagram = document.createElement("div");
      diagram.className = "mermaid";
      diagram.textContent = code.textContent;
      pre.replaceWith(diagram);
    });
    try {
      await window.mermaid.run({ nodes: container.querySelectorAll(".mermaid") });
    } catch (error) {
      console.warn("Mermaid rendering failed", error);
    }
  }
}

function resolveUrl(path) {
  if (!path) return "";

  const rawPath = String(path).trim();
  if (/^(?:[a-z]+:)?\/\//i.test(rawPath)) {
    return rawPath;
  }
  // Already an absolute site path (e.g. produced by Jekyll's relative_url).
  if (rawPath.startsWith("/")) {
    try {
      return new URL(rawPath, location.origin).toString();
    } catch (error) {
      return rawPath;
    }
  }

  // Repo-root-relative paths (e.g. "Books/…/Chapter_1.md") must resolve against
  // the site baseurl, NOT the current page — the reader can live at /books/,
  // /research/, etc. where document.baseURI would otherwise mangle the path.
  const base = (siteConfig.basePath || "").replace(/\/$/, "");
  try {
    return new URL(base + "/" + rawPath.replace(/^\//, ""), location.origin).toString();
  } catch (error) {
    console.warn("Unable to resolve path", rawPath, error);
    return rawPath;
  }
}

// Replace math spans ($$…$$, \[…\], $…$, \(…\)) with inert placeholders so the
// Markdown parser cannot mangle the TeX (notably "_" subscripts). Longest/most
// specific delimiters first. The original text (delimiters included) is stashed
// in `store` and restored verbatim after Markdown runs.
function protectMath(md, store) {
  const patterns = [
    /\$\$[\s\S]+?\$\$/g,       // display $$ … $$
    /\\\[[\s\S]+?\\\]/g,        // display \[ … \]
    /\\\([\s\S]+?\\\)/g,        // inline  \( … \)
    /\$(?!\s)[^\n$]+?(?<!\s)\$/g // inline  $ … $  (no leading/trailing space)
  ];
  let out = md;
  for (const re of patterns) {
    out = out.replace(re, (m) => {
      const token = `@@MATH${store.length}@@`;
      store.push(m);
      return token;
    });
  }
  return out;
}

function restoreMath(html, store) {
  return html.replace(/@@MATH(\d+)@@/g, (_, i) => store[Number(i)] ?? "");
}

function stripFrontMatter(markdown) {
  if (markdown.startsWith("---")) {
    const match = markdown.match(/^---[\s\S]*?\n---\s*\n?/);
    if (match) {
      return markdown.slice(match[0].length);
    }
  }
  return markdown;
}

function renderLoading() {
  contentEl.innerHTML = '<div class="bookshelf-status">Loading chapter…</div>';
}

function renderError(message) {
  contentEl.innerHTML = `<div class="bookshelf-status error"><strong>Something went wrong.</strong><br>${message}</div>`;
}

function renderNavigationPlaceholder(message) {
  navigationEl.innerHTML = `<div class="bookshelf-status error">${message}</div>`;
}

function renderContentMessage(message) {
  contentEl.innerHTML = `<div class="bookshelf-status">${message}</div>`;
}

function firstChapterPath(books) {
  for (const book of books) {
    for (const section of book.sections) {
      if (section.items && section.items.length > 0) {
        return section.items[0].path;
      }
    }
  }
  return "";
}

function handleHashChange() {
  const path = decodeHash(location.hash);
  if (!path) {
    showBooksOverview();
    return;
  }
  displayChapter(path);
}

function hideBlogSection() {
  const blogSection = document.querySelector('.home-page__column--blog');
  const grid = document.querySelector('.home-page__grid');
  if (blogSection) {
    blogSection.style.display = 'none';
  }
  if (grid) {
    grid.classList.add('home-page__grid--single');
  }
}

function showBlogSection() {
  const blogSection = document.querySelector('.home-page__column--blog');
  const grid = document.querySelector('.home-page__grid');
  if (blogSection) {
    blogSection.style.display = '';
  }
  if (grid) {
    grid.classList.remove('home-page__grid--single');
  }
}

if (navigationEl && contentEl && mainEl) {
  window.addEventListener("hashchange", handleHashChange);
  document.addEventListener("DOMContentLoaded", init);
}
