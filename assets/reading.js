// Reading enhancements: auto table-of-contents from headings with scroll-spy,
// plus a top reading-progress bar. Runs on post pages.

(function () {
  var content = document.getElementById("readingContent");
  var tocNav = document.getElementById("readingToc");
  var progress = document.getElementById("readingProgress");
  if (!content) return;

  // --- Table of contents ---------------------------------------------------
  var headings = Array.prototype.slice.call(content.querySelectorAll("h2, h3"));
  var links = [];
  if (tocNav && headings.length > 1) {
    var used = {};
    headings.forEach(function (h) {
      if (!h.id) {
        var base = (h.textContent || "section").toLowerCase()
          .replace(/[^\w\s-]/g, "").trim().replace(/\s+/g, "-") || "section";
        var id = base, i = 2;
        while (used[id] || document.getElementById(id)) { id = base + "-" + i++; }
        used[id] = true;
        h.id = id;
      }
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      a.className = "reading__toc-link reading__toc-link--" + h.tagName.toLowerCase();
      a.addEventListener("click", function (e) {
        e.preventDefault();
        var target = document.getElementById(h.id);
        if (target) {
          window.scrollTo({ top: target.getBoundingClientRect().top + window.scrollY - 90, behavior: "smooth" });
          history.replaceState(null, "", "#" + h.id);
        }
      });
      tocNav.appendChild(a);
      links.push({ id: h.id, el: a, heading: h });
    });
  } else if (tocNav) {
    var wrap = tocNav.closest(".reading__toc");
    if (wrap) wrap.style.display = "none";
  }

  // --- Scroll-spy + progress bar ------------------------------------------
  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () {
      // Progress
      if (progress) {
        var docH = document.documentElement.scrollHeight - window.innerHeight;
        var pct = docH > 0 ? Math.min(100, Math.max(0, (window.scrollY / docH) * 100)) : 0;
        progress.style.width = pct + "%";
      }
      // Active heading
      if (links.length) {
        var active = links[0];
        for (var i = 0; i < links.length; i++) {
          if (links[i].heading.getBoundingClientRect().top - 100 <= 0) active = links[i];
        }
        links.forEach(function (l) { l.el.classList.toggle("is-active", l === active); });
      }
      ticking = false;
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
})();
