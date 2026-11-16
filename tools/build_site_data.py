"""Build the site data: score every training run on the same test photos, then save example detections.

    python tools/build_site_data.py --runs path/to/Trains --dataset path/to/Datasets

Needs Ultralytics 8.3 or newer (YOLO11 layers) and a CUDA GPU.
"""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "data"

# Run folder -> dataset version it was trained on, in the order they were trained.
RUNS = [
    ("Model_ori", "ori_model"),
    ("Model_1", "model1"),
    ("Model_1.5", "model1.5"),
    ("Model_2", "model2"),
    ("Model3", "model3"),
    ("Model3.2", "model3"),
]
FINAL = "Model3"  # highest test mAP50 and mAP50-95
CONF = 0.5
KEEP = ["args.yaml", "results.csv", "results.png", "PR_curve.png", "confusion_matrix_normalized.png", "val_batch0_pred.jpg"]


def best_epoch(results_csv: Path) -> dict:
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(results_csv))]
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    return {"epochs_run": len(rows), "best_epoch": int(best["epoch"]),
            "val_map50": round(float(best["metrics/mAP50(B)"]), 3),
            "val_map50_95": round(float(best["metrics/mAP50-95(B)"]), 3)}


def evaluate(runs: Path, dataset: Path) -> list[dict]:
    from ultralytics import YOLO

    # Every dataset version shares the same 257 test images and labels, so one yaml scores them all.
    data = OUT.parent.parent / "test_split.yaml"
    data.write_text(yaml.safe_dump({"path": str(dataset / "model2"), "train": "train/images", "val": "valid/images",
                                    "test": "test/images", "names": {0: "proboscis monkey"}}))
    out = []
    for run, version in RUNS:
        args = yaml.safe_load(open(runs / run / "args.yaml", encoding="utf-8"))
        m = YOLO(runs / run / "weights" / "best.pt").val(data=str(data), split="test", imgsz=640, batch=16,
                                                        device=0, plots=False, verbose=False)
        keep = ROOT / "training_results" / run
        keep.mkdir(parents=True, exist_ok=True)
        for f in KEEP:
            if (runs / run / f).exists():
                shutil.copy2(runs / run / f, keep / f)
        out.append({
            "run": run, "dataset": version,
            "train_images": len(os.listdir(dataset / version / "train" / "images")),
            "start": "COCO pre-trained" if str(args["model"]).endswith("yolo11m.pt") else
                     "from scratch" if str(args["model"]).endswith(".yaml") else "continued from Model3",
            **best_epoch(runs / run / "results.csv"),
            "test": {"precision": round(float(m.box.mp), 3), "recall": round(float(m.box.mr), 3),
                     "map50": round(float(m.box.map50), 3), "map50_95": round(float(m.box.map), 3)},
        })
        print(run, out[-1]["test"], flush=True)
    # Model3 trained with NMS IoU 0.9, which kept duplicate boxes and lowered its validation score during training.
    model3 = YOLO(runs / "Model3" / "weights" / "best.pt")
    nms = {str(iou): round(float(model3.val(data=str(data), split="val", iou=iou, device=0, plots=False,
                                            verbose=False).box.map50), 3) for iou in (0.9, 0.7)}
    data.unlink()
    shutil.copy2(runs / FINAL / "weights" / "best.pt", ROOT / "models" / "proboscis.pt")
    (ROOT / "web" / "img").mkdir(exist_ok=True)
    shutil.copy2(runs / FINAL / "results.png", ROOT / "web" / "img" / "final-results.png")
    labels = list((dataset / "model2" / "test" / "labels").glob("*.txt"))
    test = {"test_images": len(labels), "test_boxes": sum(1 for f in labels for line in f.read_text().splitlines() if line.strip())}
    return out, nms, test


def pick_examples(dataset: Path, n_correct: int = 4) -> dict:
    """Run the final model on every test photo, count misses and false boxes, and save six examples."""
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import box_iou

    model = YOLO(ROOT / "models" / "proboscis.pt")
    found = {"correct": [], "missed": [], "false box": []}
    for img in sorted((dataset / "model2" / "test" / "images").glob("*.jpg")):
        r = model.predict(img, conf=CONF, device=0, verbose=False)[0]
        h, w = r.orig_shape
        rows = [list(map(float, line.split()[1:])) for line in
                (img.parent.parent / "labels" / f"{img.stem}.txt").read_text().splitlines() if line.strip()]
        gt = torch.tensor([[(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
                           for x, y, bw, bh in rows]).reshape(-1, 4)
        iou = box_iou(gt, r.boxes.xyxy.cpu())
        # ponytail: best-IoU match without one-to-one assignment, enough to sort photos into examples
        missed = (iou.max(1).values < 0.5) if iou.shape[1] else torch.ones(len(gt), dtype=torch.bool)
        false = int((iou.max(0).values < 0.5).sum()) if iou.shape[0] else len(r.boxes)
        kind = "missed" if missed.any() else "false box" if false else "correct"
        found[kind].append({"img": img, "r": r, "gt": gt, "missed": missed, "monkeys": len(gt), "false": false})
    out = ROOT / "web" / "img" / "examples"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    picks = sorted(found["correct"], key=lambda e: -e["monkeys"])[:n_correct] +         sorted(found["missed"], key=lambda e: -e["monkeys"])[:1] + sorted(found["false box"], key=lambda e: -e["false"])[:1]
    examples = []
    for i, e in enumerate(picks, 1):
        im = e["r"].plot(line_width=2)
        for x1, y1, x2, y2 in e["gt"][e["missed"]].int().tolist():
            cv2.rectangle(im, (x1, y1), (x2, y2), (46, 138, 201), 3)  # amber: a monkey the model missed
        cv2.imwrite(str(out / f"{i}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
        examples.append({"file": f"img/examples/{i}.jpg", "monkeys": e["monkeys"],
                         "missed": int(e["missed"].sum()), "false_boxes": e["false"]})
    return {"examples": examples, "photos": {k: len(v) for k, v in found.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True, help="the Trains folder with one sub-folder per run")
    ap.add_argument("--dataset", type=Path, required=True, help="the Datasets folder with every dataset version")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs, nms, test = evaluate(a.runs, a.dataset)
    ex = pick_examples(a.dataset)
    (OUT / "site.json").write_text(json.dumps({"final": FINAL, "conf": CONF, **test, "runs": runs,
                                               "model3_val_map50_by_nms_iou": nms, **ex}, indent=1))
    print("done:", FINAL, ex["photos"])


if __name__ == "__main__":
    main()
