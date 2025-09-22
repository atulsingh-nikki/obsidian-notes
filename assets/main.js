const state = {
  books: [],
  activePath: null,
};

const navigationEl = document.getElementById("book-navigation");
const contentEl = document.getElementById("book-content");
const mainEl = document.getElementById("main");
const navRegistry = new Map();

if (window.marked) {
  window.marked.setOptions({
    breaks: false,
    gfm: true,
    mangle: false,
    headerIds: true,
  });
}

async function init() {
  try {
    const books = await loadBooks();
    state.books = books;
    if (books.length === 0) {
      renderNavigationPlaceholder("No learning collection directories were found. Add folders that start with 'Book_'.");
      renderContentMessage("No content to display yet.");
      return;
    }

    buildNavigation(books);
    const initialPath = decodeHash(location.hash) || firstChapterPath(books);
    if (initialPath) {
      // Use replaceState so that the initial load does not add an extra entry.
      updateHash(initialPath, { replace: true });
      await displayChapter(initialPath);
    }
  } catch (error) {
    console.error(error);
    renderNavigationPlaceholder("Unable to load learning collection metadata. Ensure books.json exists.");
    renderError(String(error));
  }
}

async function loadBooks() {
  const response = await fetch("books.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`books.json could not be loaded (HTTP ${response.status})`);
  }
  return response.json();
}

function buildNavigation(books) {
  navigationEl.innerHTML = "";
  navRegistry.clear();

  books.forEach((book, bookIndex) => {
    const bookContainer = document.createElement("div");
    bookContainer.className = "bookshelf-book";

    const bookHeader = document.createElement("h3");
    bookHeader.textContent = book.title;
    bookContainer.appendChild(bookHeader);

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
    const response = await fetch(encodeURI(path));
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    let markdown = await response.text();
    markdown = stripFrontMatter(markdown);
    const html = window.marked ? window.marked.parse(markdown) : markdown;
    contentEl.innerHTML = html;
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
  if (!path) return;
  displayChapter(path);
}

window.addEventListener("hashchange", handleHashChange);
document.addEventListener("DOMContentLoaded", init);
