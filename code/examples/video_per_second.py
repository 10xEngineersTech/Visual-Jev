"""Run the released adapter on every second of a video, one decision per second.

Each whole second of the source video is scored as its own 1-second window: the
dashcam question set from the Colab prototype is asked against the frames in
that second, and the decision probabilities are written to one JSON file per
second (`sec_0000.json`, `sec_0001.json`, ...) plus a combined `summary.json`.

Run from the repository root:
    python code/examples/video_per_second.py \
        --video driving_cam.mp4 --out out/video_seconds --device mps

`--fps` sets how many frames each 1-second window carries (default 2). With
`--fps 1` each window is a single frame and the image path is used; otherwise
the frames are passed to the video path, so the prompt is built once per window
and every question is answered from the same shared visual context.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# The dashcam questions from the Colab prototype: what to do next, and whether
# anything is directly ahead in this lane.
DRIVING_QUESTIONS = {
    "action": {
        "instructions": "What should the car do next?",
        "criteria": {"accelerate": None, "brake": None,
                     "change_lane": None, "maintain_speed": None},
    },
    "obstacle_present": {
        "instructions": "Is there an obstacle (vehicle, pedestrian, or object) "
                        "directly ahead in this lane?",
        "criteria": {"yes": None, "no": None},
    },
}


def build_question(instructions: str, criteria: dict):
    """A model question dict plus the ordered option ids, from a spec entry.

    Mirrors the notebook's `_build_question`: an option whose description is
    null is used verbatim, otherwise it is rendered as "id: description".
    """
    option_ids, candidates = [], []
    for option_id, description in criteria.items():
        option_ids.append(option_id)
        candidates.append(option_id if not description else f"{option_id}: {description}")
    question = {"qtype": "choice", "instruction": instructions, "candidates": candidates}
    return question, option_ids


def questions_from_spec(spec: dict):
    """(qids, questions, option_id_lists) for a {qid: {instructions, criteria}} spec."""
    qids = list(spec)
    questions, option_lists = [], []
    for qid in qids:
        question, option_ids = build_question(spec[qid]["instructions"], spec[qid]["criteria"])
        questions.append(question)
        option_lists.append(option_ids)
    return qids, questions, option_lists


def iter_second_windows(path: str, frames_per_second: float):
    """Yield (second_index, [rgb frames]) for every whole second of the video.

    One sequential pass: frames are bucketed by their timestamp and thinned to
    at most `frames_per_second` per bucket, so a window stays cheap and no
    random seeking is needed. A second that yields no sampled frame is skipped.
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / frames_per_second))
    window: list = []
    second, idx = 0, 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            current = int(idx / src_fps)
            if current != second:
                if window:
                    yield second, window
                second, window = current, []
            if idx % step == 0:
                window.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            idx += 1
    finally:
        cap.release()
    if window:
        yield second, window


def score_window(model, frames, questions, option_lists, qids) -> dict:
    """Ask every question about one window and return {qid: {prediction, probabilities}}."""
    import torch
    from PIL import Image

    with torch.inference_mode():
        if len(frames) == 1:
            group = model.prepare_group(image=Image.fromarray(frames[0]), questions=questions)
        else:
            group = model.prepare_group(video=frames, questions=questions)
        run = model.run_independent if len(questions) == 1 else model.run_prefix_share_batch
        output = run(group)

    results = {}
    for i, qid in enumerate(qids):
        logits = output["lm_option_logits"][i, : group.n_options[i]]
        probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
        option_ids = option_lists[i]
        results[qid] = {
            "prediction": option_ids[max(range(len(probs)), key=probs.__getitem__)],
            "probabilities": dict(zip(option_ids, probs)),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="path to a local video")
    parser.add_argument("--out", required=True, help="directory for the per-second JSON files")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="frames sampled per 1-second window (1 uses the image path)")
    parser.add_argument("--limit-seconds", type=int, default=0,
                        help="stop after this many seconds (0 = whole video)")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default="guanxuyu/visual-jev-4b-answer-sft")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                        help="auto selects CUDA first, then Apple MPS, then CPU")
    parser.add_argument("--max-pixels", type=int, default=200704,
                        help="image pixel budget; lower it if unified memory is tight")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")

    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from peft import PeftModel

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vdm.models.vdm_model import VDM

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but this PyTorch installation has no available CUDA device")
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested, but Apple MPS is not available in this PyTorch installation")
    dtype = torch.float16 if device == "mps" else torch.bfloat16

    model = VDM(args.model, device=device, dtype=dtype, with_heads=False)
    print(f"Using {device} with {dtype}")
    model.processor.image_processor.max_pixels = args.max_pixels
    model.backbone = PeftModel.from_pretrained(model.backbone, args.adapter).eval()

    qids, questions, option_lists = questions_from_spec(DRIVING_QUESTIONS)

    os.makedirs(args.out, exist_ok=True)
    summary = []
    t_start = time.time()
    for sec, frames in iter_second_windows(args.video, args.fps):
        if args.limit_seconds and len(summary) >= args.limit_seconds:
            break
        t0 = time.time()
        results = score_window(model, frames, questions, option_lists, qids)
        elapsed = time.time() - t0
        record = {
            "video": args.video,
            "second": sec,
            "start_s": float(sec),
            "end_s": float(sec + 1),
            "n_frames": len(frames),
            "device": device,
            "latency_s": round(elapsed, 3),
            "results": results,
        }
        with open(os.path.join(args.out, f"sec_{sec:04d}.json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        summary.append(record)
        print(f"[{sec:>4}s] {len(frames)}f  {elapsed * 1000:6.0f} ms  "
              + "  ".join(f"{q}={results[q]['prediction']}" for q in qids), flush=True)

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {len(summary)} per-second files + summary.json -> {args.out} "
          f"({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
