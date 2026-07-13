(() => {
  const $ = (selector) => document.querySelector(selector);
  const escape = (value) =>
    String(value ?? "").replace(
      /[&<>'"]/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          "'": "&#39;",
          '"': "&quot;",
        })[c],
    );
  const dashboardConfig = document.getElementById("dashboard-config");
  const movieConfig = document.getElementById("movie-config");
  const userId = Number(
    dashboardConfig?.dataset.userId ?? movieConfig?.dataset.userId ?? 0,
  );
  const weights = () => ({
    collaborative: +$("#collaborative").value,
    genre: +$("#genre").value,
    language: +$("#language").value,
  });
  const showPipeline = async (steps) => {
    const list = $("#pipeline");
    if (!list) return;
    list.innerHTML = steps.map((step) => `<li>${escape(step)}</li>`).join("");
    $("#pipeline-state").textContent = "Processing…";
    for (const item of [...list.children]) {
      item.classList.add("active");
      await new Promise((resolve) => setTimeout(resolve, 90));
      item.classList.replace("active", "complete");
    }
    $("#pipeline-state").textContent = "Finished";
  };
  const render = (data, model = "localized") => {
    const result = data.results[model];
    if (!result) return;
    $("#results").hidden = false;
    $("#generated-at").textContent =
      `Session #${data.session_id} · ${new Date(data.generated_at).toLocaleTimeString()}`;
    $("#model-tabs").innerHTML = Object.entries(data.results)
      .map(
        ([key, value]) =>
          `<button data-model="${key}" class="${key === model ? "active" : ""}">${escape(value.label)}</button>`,
      )
      .join("");
    $("#model-tabs")
      .querySelectorAll("button")
      .forEach(
        (button) => (button.onclick = () => render(data, button.dataset.model)),
      );
    $("#recommendation-cards").innerHTML = result.recommendations
      .map(
        (item) =>
          `<article class="movie-card rec-card" data-movie="${item.movieId}"><div class="poster ${item.poster_url ? "" : "poster-fallback"}">${item.poster_url ? `<img src="${escape(item.poster_url)}" alt="${escape(item.title)} poster">` : `<span>${escape(item.title)}</span>`}<span class="rank">#${item.rank}</span></div><div class="movie-copy"><h3>${escape(item.title)}</h3><div class="meta">${escape(item.year || "—")} · ${escape(item.language)}</div><div class="tags">${item.genres
            .split("|")
            .slice(0, 2)
            .map((g) => `<span>${escape(g)}</span>`)
            .join(
              "",
            )}</div><div class="score">Score ${item.score.toFixed(4)}</div><p class="reason">${escape(item.reason)}</p></div></article>`,
      )
      .join("");
    $("#recommendation-cards")
      .querySelectorAll(".rec-card")
      .forEach(
        (card) =>
          (card.onclick = () =>
            (location.href = `/movie/${card.dataset.movie}?user_id=${userId}`)),
      );
    const metrics = data.metrics[model];
    $("#metrics-grid").innerHTML = Object.entries(metrics)
      .map(
        ([key, value]) =>
          `<div class="metric"><span>${escape(key.replaceAll("_", " "))}</span><b>${typeof value === "number" ? value.toFixed(key.includes("diversity") ? 0 : 4) : escape(value)}</b></div>`,
      )
      .join("");
    const changes = data.changes[model];
    $("#changes").innerHTML = [
      ["New movies", changes.new],
      ["Removed", changes.removed],
      [
        "Rank movement",
        changes.moved.map((x) => `${x.title}: #${x.from} → #${x.to}`),
      ],
    ]
      .map(
        ([title, items]) =>
          `<div class="change-group"><b>${title}</b>${
            items.length
              ? `<ul>${items
                  .slice(0, 5)
                  .map((x) => `<li>${escape(x)}</li>`)
                  .join("")}</ul>`
              : '<span class="caption">No change</span>'
          }</div>`,
      )
      .join("");
  };
  const generate = async () => {
    const response = await fetch("/api/recommendations/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, ...weights() }),
    });
    const data = await response.json();
    await showPipeline(data.pipeline);
    render(data);
  };
  if ($("#generate")) {
    ["collaborative", "genre", "language"].forEach(
      (id) =>
        ($("#" + id).oninput = () =>
          ($("#" + id + "-value").textContent = (+$("#" + id).value).toFixed(
            2,
          ))),
    );
    $("#generate").onclick = generate;
    const saved = sessionStorage.getItem("recosys-result");
    if (saved) {
      sessionStorage.removeItem("recosys-result");
      const data = JSON.parse(saved);
      showPipeline(data.pipeline).then(() => render(data));
    } else generate();
  }
  if ($("#star-picker") && movieConfig) {
    let selected = Number(movieConfig?.dataset.currentRating || 0);
    const stars = [...$("#star-picker").children];
    const draw = () =>
      stars.forEach((star) =>
        star.classList.toggle("selected", +star.dataset.rating <= selected),
      );
    draw();
    stars.forEach(
      (star) =>
        (star.onclick = () => {
          selected = +star.dataset.rating;
          draw();
          $("#rating-label").textContent = `Selected: ${selected} / 5`;
          $("#submit-rating").disabled = false;
        }),
    );
    $("#submit-rating").onclick = async () => {
      const progress = $("#rate-progress");
      progress.hidden = false;
      progress.innerHTML =
        '<div class="eyebrow">Saving rating</div><h2>Updating your recommendations…</h2>';
      const response = await fetch(
        `/api/movies/${movieConfig.dataset.movieId}/rate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: userId,
            rating: selected,
            collaborative: 0.5,
            genre: 0.18,
            language: 0.12,
          }),
        },
      );
      const data = await response.json();
      progress.innerHTML = `<div class="eyebrow">Rating saved</div><h2>Recommendations regenerated</h2><p class="caption">Your rating is persisted in SQLite. Opening the lab will show the new Top-10 lists, metrics and before/after comparison.</p>`;
      sessionStorage.setItem("recosys-result", JSON.stringify(data));
      setTimeout(() => (location.href = `/dashboard/${userId}`), 800);
    };
  }
})();
