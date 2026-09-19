// Research vault index: topic + title filtering. Each note links to its own
// rendered page (via the research-note layout), so no client-side reader here.

(function () {
  const emptyEl = document.getElementById("researchEmpty");
  const searchEl = document.getElementById("researchSearch");
  const topicChips = document.querySelectorAll("#researchTopics .chip");
  const items = Array.from(document.querySelectorAll(".research-item"));

  let activeTopic = "all";
  let query = "";

  function apply() {
    let visible = 0;
    items.forEach((item) => {
      const topic = item.getAttribute("data-topic");
      const title = item.getAttribute("data-title") || "";
      const show = (activeTopic === "all" || topic === activeTopic) && title.includes(query);
      item.style.display = show ? "" : "none";
      if (show) visible++;
    });
    document.querySelectorAll(".research-year").forEach((sec) => {
      const any = Array.from(sec.querySelectorAll(".research-item")).some((i) => i.style.display !== "none");
      sec.style.display = any ? "" : "none";
    });
    if (emptyEl) emptyEl.hidden = visible !== 0;
  }

  topicChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      activeTopic = chip.getAttribute("data-topic");
      topicChips.forEach((c) => c.classList.toggle("is-active", c === chip));
      apply();
    });
  });

  if (searchEl) {
    searchEl.addEventListener("input", () => {
      query = searchEl.value.toLowerCase().trim();
      apply();
    });
  }
})();
