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

  const METRIC_LABELS = {
    language_diversity: "Language diversity",
    genre_diversity: "Genre diversity",
    filter_bubble_score: "Filter bubble score",
    preference_escape_at_10: "Preference escape@10",
  };
  const INTEGER_METRICS = new Set(["language_diversity", "genre_diversity"]);

  const postJSON = async (url, body) => {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  };

  const dashboardConfig = document.getElementById("dashboard-config");
  const movieConfig = document.getElementById("movie-config");
  const userId = Number(
    dashboardConfig?.dataset.userId ?? movieConfig?.dataset.userId ?? 0,
  );

  const renderPipeline = (stages) => {
    const list = $("#pipeline");
    if (!list) return;
    const total = stages.reduce((sum, s) => sum + s.ms, 0);
    list.innerHTML = stages
      .map(
        (s) =>
          `<li class="complete"><span>${escape(s.stage)}</span><b>${s.ms.toFixed(1)} ms</b></li>`,
      )
      .join("");
    $("#pipeline-state").textContent = `${total.toFixed(0)} ms total`;
  };

  const formatWeights = (weights) => {
    const entries = Object.entries(weights || {});
    if (!entries.length) return "no weighted components";
    return entries
      .map(([key, value]) => `${escape(key)} ${value.toFixed(3)}`)
      .join(" · ");
  };

  const render = (data, model = "localized") => {
    const result = data.results[model];
    if (!result) return;
    $("#results").hidden = false;
    $("#generated-at").textContent =
      `Session #${data.session_id} · ${data.trigger} · ${new Date(data.generated_at).toLocaleTimeString()}`;

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

    $("#model-description").innerHTML =
      `${escape(result.description)} <b>Weights:</b> ${formatWeights(result.weights)}.`;

    const metrics = data.metrics[model] || {};
    $("#metrics-grid").innerHTML = Object.entries(metrics)
      .map(([key, value]) => {
        const label = METRIC_LABELS[key] ?? key.replaceAll("_", " ");
        const shown = INTEGER_METRICS.has(key)
          ? value.toFixed(0)
          : value.toFixed(3);
        return `<div class="metric"><span>${escape(label)}</span><b>${shown}</b></div>`;
      })
      .join("");

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

    const changes = data.changes[model];
    $("#changes").innerHTML = [
      ["New in top 10", changes.new],
      ["Dropped out", changes.removed],
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

  let generating = false;
  const generate = async () => {
    if (generating) return;
    generating = true;
    const button = $("#generate");
    if (button) {
      button.disabled = true;
      button.textContent = "Scoring six models…";
    }
    $("#pipeline-state").textContent = "Processing…";
    try {
      const data = await postJSON("/api/recommendations/generate", {
        user_id: userId,
      });
      renderPipeline(data.pipeline);
      render(data);
    } catch (error) {
      $("#pipeline-state").textContent = "Failed";
      $("#pipeline").innerHTML =
        `<li class="failed">${escape(error.message)}</li>`;
    } finally {
      generating = false;
      if (button) {
        button.disabled = false;
        button.textContent = "Regenerate recommendations";
      }
    }
  };

  if ($("#jump-form")) {
    $("#jump-form").onsubmit = (event) => {
      event.preventDefault();
      const id = $("#jump-user-id").value.trim();
      if (id) location.href = `/dashboard/${id}`;
    };
  }

  if ($("#generate")) {
    $("#generate").onclick = generate;
    const saved = sessionStorage.getItem("recosys-result");
    if (saved) {
      sessionStorage.removeItem("recosys-result");
      const data = JSON.parse(saved);
      renderPipeline(data.pipeline);
      render(data);
    }
  }

  if ($("#star-picker") && movieConfig) {
    let selected = Number(movieConfig.dataset.currentRating || 0);
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
      const button = $("#submit-rating");
      button.disabled = true;
      progress.hidden = false;
      progress.innerHTML =
        '<div class="eyebrow">Saving rating</div><h2>Re-scoring all six models…</h2>';
      try {
        const data = await postJSON(
          `/api/movies/${movieConfig.dataset.movieId}/rate`,
          { user_id: userId, rating: selected },
        );
        progress.innerHTML =
          '<div class="eyebrow">Rating saved</div><h2>Recommendations regenerated</h2><p class="caption">Returning to the lab with the before/after comparison.</p>';
        sessionStorage.setItem("recosys-result", JSON.stringify(data));
        setTimeout(() => (location.href = `/dashboard/${userId}`), 800);
      } catch (error) {
        button.disabled = false;
        progress.innerHTML = `<div class="eyebrow">Failed</div><p class="caption">${escape(error.message)}</p>`;
      }
    };
  }
})();
