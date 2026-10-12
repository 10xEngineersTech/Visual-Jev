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

Outputs, all under `--out`:

    sec_0000.json ...  one record per scored second
    summary.json       the same records as one list
    overlay.mp4        the source video re-encoded with each second's answers
                       drawn over it (`--out-video` moves it elsewhere)

While it runs, progress and log lines go to stderr: a tqdm bar tracks the
seconds being scored and the frames being re-encoded into the overlay, and log
lines report device selection, model download/load, adapter attach, and every
second's decision. tqdm arrives with transformers, so it is already installed
wherever this script can run.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from tqdm import tqdm

# The dashcam question set, grouped the way a driver-assistance stack reads a
# scene: the decision and its urgency first, then the forward-collision cues
# behind it, the signals and signs that constrain it, and the conditions that
# degrade it. Each question is one line in the overlay, so the panel grows with
# this dict; drop entries here to trade coverage for latency and screen space.
# Options with a description are rendered as "id: description" to disambiguate
# them; the id alone is what lands in the JSON and on the overlay.
DRIVING_QUESTIONS = {
    # -- decision ----------------------------------------------------------
    "action": {
        "instructions": "What should the car do next?",
        "criteria": {"accelerate": None, "brake": None,
                     "change_lane": None, "maintain_speed": None},
    },
    "warning_level": {
        "instructions": "How urgently should the driver be warned about a hazard right now?",
        "criteria": {"none": "no hazard needing attention",
                     "caution": "a developing hazard the driver should watch",
                     "urgent": "an imminent hazard requiring immediate action"},
    },
    # -- forward collision ---------------------------------------------------
    "obstacle_present": {
        "instructions": "Is there an obstacle (vehicle, pedestrian, or object) "
                        "directly ahead in this lane?",
        "criteria": {"yes": None, "no": None},
    },
    "lead_vehicle_gap": {
        "instructions": "How far ahead is the nearest vehicle in this lane?",
        "criteria": {"none": "no vehicle ahead in this lane",
                     "far": "a comfortable following distance",
                     "close": "closer than a safe following distance",
                     "very_close": "a collision is likely without braking"},
    },
    "lead_vehicle_braking": {
        "instructions": "Are the brake lights of the vehicle directly ahead turned on?",
        "criteria": {"yes": None, "no": None, "no_vehicle": None},
    },
    "vehicle_cut_in": {
        "instructions": "Is a vehicle from an adjacent lane merging or cutting into this lane ahead?",
        "criteria": {"yes": None, "no": None},
    },
    "vulnerable_road_user": {
        "instructions": "Where is the nearest pedestrian, cyclist, or motorcyclist?",
        "criteria": {"none": "none visible",
                     "roadside": "on the sidewalk or shoulder, not in the path",
                     "in_path": "in or entering the car's path"},
    },
    "emergency_vehicle": {
        "instructions": "Is there an emergency vehicle (ambulance, fire truck, or police car) "
                        "with flashing lights?",
        "criteria": {"yes": None, "no": None},
    },
    # -- signals, signs, and lane -------------------------------------------
    "traffic_light": {
        "instructions": "What color is the traffic light that controls this lane?",
        "criteria": {"none": "no traffic light applies", "red": None,
                     "yellow": None, "green": None},
    },
    "traffic_sign": {
        "instructions": "Which traffic sign facing the car is most relevant right now?",
        "criteria": {"none": None, "stop": None, "yield": None, "speed_limit": None,
                     "pedestrian_crossing": None, "other": None},
    },
    "lane_position": {
        "instructions": "Where is the car positioned within its lane?",
        "criteria": {"centered": None, "drifting_left": None, "drifting_right": None},
    },
    "construction_zone": {
        "instructions": "Is the car in or approaching a construction or work zone "
                        "(cones, barriers, or workers)?",
        "criteria": {"yes": None, "no": None},
    },
    # -- conditions ----------------------------------------------------------
    "road_surface": {
        "instructions": "What is the condition of the road surface?",
        "criteria": {"dry": None, "wet": None, "snow_or_ice": None},
    },
    "visibility": {
        "instructions": "How good is the visibility of the road ahead?",
        "criteria": {"clear": None,
                     "reduced": "rain, fog, dusk, or glare limits the view",
                     "poor": "darkness or heavy weather hides the road ahead"},
    },
}


log = logging.getLogger("video_per_second")


class _TqdmHandler(logging.StreamHandler):
    """Write log lines through `tqdm.write` so they never smear a live bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:  # logging must not raise into the caller
            self.handleError(record)


def configure_logging() -> None:
    """Timestamped INFO lines on stderr, interleaved cleanly with the bars."""
    handler = _TqdmHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                           datefmt="%H:%M:%S"))
    log.setLevel(logging.INFO)
    log.handlers = [handler]
    log.propagate = False


def open_video(path: str):
    """A cv2 capture for `path`, or a readable exit if it will not open."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {path}")
    return cap


def video_info(path: str) -> tuple[float, int]:
    """(source fps, frame count) read from the container; the count is 0 if unknown."""
    import cv2

    cap = open_video(path)
    try:
        return cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


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


def sample_step(src_fps: float, frames_per_second: float) -> int:
    """Keep every `step`-th frame -- the single place sampling density is decided.

    Read by both the window iterator and the progress-bar total, so the bar
    counts exactly the windows the iterator will yield.
    """
    return max(1, round(src_fps / frames_per_second))


def iter_second_windows(path: str, frames_per_second: float):
    """Yield (second_index, [rgb frames]) for every whole second of the video.

    One sequential pass: frames are bucketed by their timestamp and thinned to
    at most `frames_per_second` per bucket, so a window stays cheap and no
    random seeking is needed. A second that yields no sampled frame is skipped.
    """
    import cv2

    cap = open_video(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = sample_step(src_fps, frames_per_second)
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


def draw_answers(frame, record: dict, qids) -> None:
    """Draw one second's answers into the top-left corner of a BGR frame, in place.

    The panel is a darkened rectangle so white-on-road stays readable at any
    resolution. The first row carries the timing; the answers below it flow
    down as many columns as fit the frame width, so a long question set widens
    the panel instead of letting it run down the frame.
    """
    import cv2

    header = f"{record['second']}s  {record['n_frames']}f  {record['latency_s'] * 1000:.0f} ms"
    answers = []
    for qid in qids:
        result = record["results"][qid]
        prediction = result["prediction"]
        answers.append(f"{qid}: {prediction} {result['probabilities'][prediction]:.2f}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_h, frame_w = frame.shape[:2]
    scale = min(0.8, max(0.35, min(frame_h, frame_w) / 360 * 0.3))
    thickness = max(1, round(scale * 1.6))
    pad = int(round(scale * 8))
    gap = int(round(scale * 24))  # between answer columns

    def text_w(line: str) -> int:
        return cv2.getTextSize(line, font, scale, thickness)[0][0]

    # Measured rather than fixed, so rows never touch whatever font cv2 renders.
    (_, glyph_h), descent = cv2.getTextSize("Ag_", font, scale, thickness)
    line_h = glyph_h + descent + max(2, pad // 2)

    col_w = max(map(text_w, answers)) + gap
    n_cols = max(1, min(len(answers), (frame_w - 2 * pad + gap) // col_w))
    n_rows = -(-len(answers) // n_cols)
    n_cols = -(-len(answers) // n_rows)  # rebalance so the last column is not near-empty

    panel_w = min(frame_w, max(text_w(header), n_cols * col_w - gap) + 2 * pad)
    panel_h = min(frame_h, line_h * (n_rows + 1) + pad)
    panel = frame[:panel_h, :panel_w]
    panel[:] = (panel * 0.4).astype(panel.dtype)

    baseline = pad + glyph_h
    cv2.putText(frame, header, (pad, baseline), font, scale, (255, 255, 255), thickness,
                cv2.LINE_AA)  # BGR: white header, yellow answers
    for i, line in enumerate(answers):
        col, row = divmod(i, n_rows)
        cv2.putText(frame, line, (pad + col * col_w, baseline + (row + 1) * line_h), font, scale,
                    (0, 255, 255), thickness, cv2.LINE_AA)


def write_overlay_video(src_path: str, records: list[dict], out_path: str, qids) -> int:
    """Re-encode `src_path` with every second's answers drawn over the frames.

    A second sequential pass at the source frame rate, stopping after the last
    scored second, so a run cut short by `--limit-seconds` yields a
    correspondingly short video. `records` is already in memory, so the pass
    holds nothing but the frame it is drawing on; the extra decode costs far
    less than the inference that produced the records. Returns frames written.
    """
    import cv2

    by_second = {record["second"]: record for record in records}
    last_second = max(by_second)
    cap = open_video(src_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            raise SystemExit(f"could not open a video writer for: {out_path}")
        total = min(src_frames, int(round((last_second + 1) * fps))) if src_frames else None
        written, idx = 0, 0
        try:
            with tqdm(total=total, desc="overlay", unit="f") as bar:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    second = int(idx / fps)
                    if second > last_second:
                        break
                    record = by_second.get(second)
                    if record is not None:
                        draw_answers(frame, record, qids)
                    writer.write(frame)
                    written += 1
                    idx += 1
                    bar.update(1)
        finally:
            writer.release()
    finally:
        cap.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="path to a local video")
    parser.add_argument("--out", required=True, help="directory for the per-second JSON files")
    parser.add_argument("--out-video", default="",
                        help="path for the annotated video (default: <out>/overlay.mp4)")
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

    configure_logging()
    if args.fps <= 0:
        parser.error("--fps must be positive")

    # Read the container up front: a bad path fails before the model download,
    # and the frame count is what makes the progress bar determinate.
    src_fps, src_frames = video_info(args.video)
    if src_frames:
        # Bucket the frames the reader will keep, without decoding: `--fps`
        # below the source rate can leave a second with no sampled frame.
        step = sample_step(src_fps, args.fps)
        total_seconds = len({int(idx / src_fps) for idx in range(0, src_frames, step)})
    else:
        total_seconds = None
    if args.limit_seconds:
        total_seconds = (args.limit_seconds if total_seconds is None
                         else min(total_seconds, args.limit_seconds))
    log.info("video %s: %d frames @ %.3g fps%s", args.video, src_frames, src_fps,
             "" if not src_frames else f" ({src_frames / src_fps:.1f}s)")

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
        log.info("--device auto resolved to %s", device)
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but this PyTorch installation has no available CUDA device")
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested, but Apple MPS is not available in this PyTorch installation")
    dtype = torch.float16 if device == "mps" else torch.bfloat16

    log.info("loading %s on %s with %s (the first run downloads the weights)",
             args.model, device, dtype)
    t_load = time.time()
    model = VDM(args.model, device=device, dtype=dtype, with_heads=False)
    model.processor.image_processor.max_pixels = args.max_pixels
    model.backbone = PeftModel.from_pretrained(model.backbone, args.adapter).eval()
    log.info("model + adapter %s ready in %.1fs (max_pixels=%d)",
             args.adapter, time.time() - t_load, args.max_pixels)

    qids, questions, option_lists = questions_from_spec(DRIVING_QUESTIONS)
    log.info("asking %s once per second", ", ".join(qids))

    os.makedirs(args.out, exist_ok=True)
    log.info("writing per-second records to %s", args.out)
    summary = []
    t_start = time.time()
    with tqdm(total=total_seconds, desc="scoring", unit="win") as bar:
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
            bar.update(1)
            log.info("[%4ds] %df  %6.0f ms  %s", sec, len(frames), elapsed * 1000,
                     "  ".join(f"{q}={results[q]['prediction']}" for q in qids))

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("wrote %d per-second files + summary.json -> %s (%.0fs total)",
             len(summary), args.out, time.time() - t_start)

    if not summary:
        log.warning("no second produced a window; nothing to overlay")
        return
    out_video = args.out_video or os.path.join(args.out, "overlay.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)
    log.info("writing the annotated video to %s", out_video)
    frames_written = write_overlay_video(args.video, summary, out_video, qids)
    log.info("wrote %s: %d frames at the source frame rate", out_video, frames_written)


if __name__ == "__main__":
    main()
