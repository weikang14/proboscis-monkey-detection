"""Build the site data: score every training run on the same test images, then render the demo clip.

    python tools/build_site_data.py --runs path/to/Trains --dataset path/to/Datasets --video path/to/sample.mp4

Needs Ultralytics 8.3 or newer (YOLO11 layers) and ffmpeg (FFMPEG env var or on PATH).
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
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


def render_demo(video: Path, seconds: int = 30) -> dict:
    """Detect on every frame, keep a per-second timeline, and write the busiest window as an H.264 clip."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(ROOT / "models" / "proboscis.pt")
    detect = lambda frame: model.predict(frame, conf=CONF, device=0, verbose=False)[0]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    hits = []
    while (frame := cap.read()[1]) is not None:
        hits.append(len(detect(frame).boxes))
    n = int(seconds * fps)
    start = max(range(0, max(1, len(hits) - n), int(fps)), key=lambda i: sum(h > 0 for h in hits[i:i + n]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ff = subprocess.Popen([os.environ.get("FFMPEG", "ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo",
                           "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                           "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / "demo.mp4")],
                          stdin=subprocess.PIPE)
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        ff.stdin.write(detect(frame).plot(line_width=2).tobytes())
    ff.stdin.close()
    ff.wait()
    per_second = [max(hits[i:i + int(fps)]) for i in range(0, len(hits), int(fps))]
    return {"duration_s": round(len(hits) / fps), "clip_start_s": round(start / fps), "clip_s": seconds,
            "seconds_with_monkeys": sum(c > 0 for c in per_second), "per_second": per_second}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True, help="the Trains folder with one sub-folder per run")
    ap.add_argument("--dataset", type=Path, required=True, help="the Datasets folder with every dataset version")
    ap.add_argument("--video", type=Path, required=True, help="footage for the demo clip")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs, nms, test = evaluate(a.runs, a.dataset)
    demo = render_demo(a.video)
    (OUT / "site.json").write_text(json.dumps({"final": FINAL, "conf": CONF, **test, "runs": runs, "model3_val_map50_by_nms_iou": nms, "demo": demo}, indent=1))
    print("done:", FINAL, demo["seconds_with_monkeys"], "of", demo["duration_s"], "s with monkeys")


if __name__ == "__main__":
    main()
