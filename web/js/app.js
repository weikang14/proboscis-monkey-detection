// Fills the page from data/site.json, written by tools/build_site_data.py.
const GREEN = "#2E7D4F", AMBER = "#C98A2E", INK2 = "#56655B", RULE = "#DCE3DE";
const f3 = (x) => x.toFixed(3);

export function facts(site) {
  const runs = Object.fromEntries(site.runs.map((r) => [r.run, r]));
  const test = site.runs.map((r) => r.test.map50);
  const scratch = site.runs.filter((r) => r.start === "from scratch").map((r) => r.best_epoch);
  return {
    ...site, runs, conf: site.conf,
    nms_hi: f3(site.model3_val_map50_by_nms_iou["0.9"]), nms_lo: f3(site.model3_val_map50_by_nms_iou["0.7"]),
    test_min: f3(Math.min(...test)), test_max: f3(Math.max(...test)),
    scratch_epochs: `${Math.min(...scratch)} to ${Math.max(...scratch)}`,
    video_pcts: ((p) => `${p[0]} of the frames of the first clip, ${p[1]} of the second and ${p[2]} of the third`)(
      site.video.map((v) => `${Math.round((100 * v.frames_boxed) / v.frames)}%`)),
  };
}

export function caption({ monkeys, missed, false_boxes }) {
  const extra = false_boxes ? `, plus ${false_boxes} extra box${false_boxes > 1 ? "es" : ""}` : "";
  if (missed) return `Missed ${missed} of ${monkeys} monkeys${extra}`;
  return (monkeys === 1 ? "Found the monkey" : `Found all ${monkeys} monkeys`) + extra;
}

const get = (obj, path) => path.split(".").reduce((o, k) => o?.[k], obj);

export function fill(root, site) {
  const f = facts(site);
  root.querySelectorAll("[data-v]").forEach((el) => { el.textContent = get(f, el.dataset.v) ?? "?"; });
  const t = f.runs[site.final].test;
  root.querySelector("#stats").innerHTML = [["mAP50", t.map50], ["mAP50-95", t.map50_95], ["Precision", t.precision], ["Recall", t.recall]]
    .map(([k, v]) => `<div class="stat"><b>${f3(v)}</b><span>${k}</span></div>`).join("");
  root.querySelector("#video-credits").innerHTML = site.video.map((v) => `<a href="${v.page}">${v.title}</a>`).join(", ");
  root.querySelector("#examples").innerHTML = site.examples.map((e) => `
    <figure><a href="${e.file}"><img src="${e.file}" alt="${caption(e)}" loading="lazy"></a><figcaption>${caption(e)}</figcaption></figure>`).join("");
  root.querySelector("#runs-table tbody").innerHTML = site.runs.map((r) => `
    <tr class="${r.run === site.final ? "final" : ""}"><th scope="row">${r.run}</th><td>${r.train_images.toLocaleString("en")}</td><td>${r.start}</td><td>${r.best_epoch} of ${r.epochs_run}</td>
    <td>${f3(r.val_map50)}</td><td>${f3(r.test.map50)}</td><td>${f3(r.test.map50_95)}</td><td>${f3(r.test.precision)}</td><td>${f3(r.test.recall)}</td></tr>`).join("");
}

function charts(site) {
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.color = INK2;
  const grid = { color: RULE };
  const dot = (color) => ({ showLine: false, pointRadius: 7, pointHoverRadius: 9, backgroundColor: color, borderColor: "#FFF", borderWidth: 2 });
  new Chart(document.getElementById("runs-chart"), {
    type: "line",
    data: {
      labels: site.runs.map((r) => r.run),
      datasets: [
        { label: "Validation mAP50, as logged", data: site.runs.map((r) => r.val_map50), ...dot(AMBER) },
        { label: "Test mAP50, re-scored", data: site.runs.map((r) => r.test.map50), ...dot(GREEN) },
      ],
    },
    options: {
      maintainAspectRatio: false,
      scales: { y: { min: 0.84, max: 0.94, grid, ticks: { callback: f3 }, title: { display: true, text: "mAP50" } }, x: { offset: true, grid: { display: false } } },
      plugins: { legend: { position: "top", align: "start", labels: { usePointStyle: true } }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${f3(c.raw)}` } } },
    },
  });
}

if (typeof document !== "undefined") {
  const site = await fetch("data/site.json").then((r) => r.json());
  fill(document, site);
  charts(site);
}
