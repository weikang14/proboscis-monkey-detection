"""The README's numbers must match the published site data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = json.loads((ROOT / "web" / "data" / "site.json").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_runs_table_matches_the_data():
    for r in SITE["runs"]:
        t = r["test"]
        cells = [f"{r['train_images']:,}", r["start"], f"{r['best_epoch']} of {r['epochs_run']}", f"{r['val_map50']:.3f}",
                 f"{t['map50']:.3f}", f"{t['map50_95']:.3f}", f"{t['precision']:.3f}", f"{t['recall']:.3f}"]
        row = next(line for line in README.splitlines() if line.startswith(f"| {r['run']}") or line.startswith(f"| **{r['run']}"))
        assert all(c in row for c in cells), (r["run"], row)


def test_final_model_is_the_best_on_test():
    best = max(SITE["runs"], key=lambda r: (r["test"]["map50_95"], r["test"]["map50"]))
    assert best["run"] == SITE["final"]
    assert f"**{SITE['final']} (final)**" in README


def test_findings_quote_the_data():
    nms, photos = SITE["model3_val_map50_by_nms_iou"], SITE["photos"]
    test = [r["test"]["map50"] for r in SITE["runs"]]
    assert f"from {nms['0.9']:.3f} to {nms['0.7']:.3f}" in README
    assert f"between {min(test):.3f} and {max(test):.3f}" in README
    assert f"on {photos['correct']} of the {SITE['test_images']} test photos, missed at least one monkey on {photos['missed']}" in README
    assert f"added an extra box on {photos['false box']} more" in README
    assert f"({SITE['test_boxes']} monkeys in the test photos)" in README


def test_video_section_quotes_the_data_and_credits_every_clip():
    p = [f"{100 * v['frames_boxed'] / v['frames']:.0f}%" for v in SITE["video"]]
    assert f"drew a box on {p[0]} of the frames of the first clip, {p[1]} of the second and {p[2]} of the third" in README
    for v in SITE["video"]:
        assert f"[{v['title']}]({v['page']})" in README
    assert "CC BY-SA 2.0" in README
