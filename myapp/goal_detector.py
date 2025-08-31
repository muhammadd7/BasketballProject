import os
import cv2
import numpy as np
from datetime import datetime
from collections import deque
import torch

# Ensure headless OpenCV has GUI stubs required by ultralytics
if not hasattr(cv2, "imshow"):
    def _cv2_noop(*args, **kwargs):
        return None
    cv2.imshow = _cv2_noop  # type: ignore[attr-defined]
    cv2.waitKey = lambda *args, **kwargs: 0  # type: ignore[attr-defined]
    cv2.destroyAllWindows = _cv2_noop  # type: ignore[attr-defined]

from ultralytics import YOLO
import shutil

# Use the shared utilities for device selection and shot logic
from utils import (
    score,
    detect_down,
    detect_up,
    in_hoop_region,
    clean_hoop_pos,
    clean_ball_pos,
    get_device,
)

# Import OCR functionality
from myapp.detection import extract_scores_from_frame, get_ocr_reader, auto_detect_scoreboard_rois

# Device selection
_DEVICE = get_device()

# Cache a single YOLO model (best.pt) to avoid reloading per request
_GLOBAL_BEST = {
    'model': None,
    'path': None,
}


def _get_best_model(model_path: str) -> YOLO:
    reload_needed = (_GLOBAL_BEST['model'] is None) or (_GLOBAL_BEST['path'] != model_path)
    if reload_needed:
        model = YOLO(model_path)
        try:
            model.fuse()  # small speed boost, safe to ignore failures
        except Exception:
            pass
        _GLOBAL_BEST['model'] = model
        _GLOBAL_BEST['path'] = model_path
    return _GLOBAL_BEST['model']


def detect_goals(
    video_path: str,
    output_dir: str,
    model_path: str = "/Users/dev/Desktop/football_project/models/large-best.pt",
    *,
    imgsz: int = 512,
    frame_stride: int = 2,
    write_tracking: bool = False,
    use_ocr: bool = False,
    score_roi: tuple = None,
    team_a_roi: tuple = None,
    team_b_roi: tuple = None,
):
    """
    Detect goals (made shots) using a single YOLO model trained for 'Basketball' and 'Basketball Hoop'.
    Counting logic is adapted from shot_detector.py and utils.py (up/down regions + score()).
    Optionally uses OCR to detect scoreboard changes.

    Args:
        video_path: Path to input video
        output_dir: Directory to save highlights
        model_path: Path to YOLO model
        imgsz: Image size for inference
        frame_stride: Process every Nth frame
        write_tracking: Save tracking visualization
        use_ocr: Use OCR to detect scoreboard
        score_roi: Region for combined score (x,y,w,h)
        team_a_roi: Region for team A score (x,y,w,h)
        team_b_roi: Region for team B score (x,y,w,h)

    Returns: (highlight_clip_name | None, total_makes:int, team_a_makes:int, team_b_makes:int)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure per-upload clip directories exist and are empty
    makes_dir = os.path.join(output_dir, 'makes')
    miss_dir = os.path.join(output_dir, 'miss')
    os.makedirs(makes_dir, exist_ok=True)
    os.makedirs(miss_dir, exist_ok=True)

    def _empty_dir(path):
        for name in os.listdir(path):
            p = os.path.join(path, name)
            try:
                if os.path.isfile(p) or os.path.islink(p):
                    os.unlink(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
            except Exception as e:
                print(f"Failed to delete {p}: {e}")

    _empty_dir(makes_dir)
    _empty_dir(miss_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Can't open video")
        return None, 0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    highlight_name = f"highlights_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    highlight_path = os.path.join(output_dir, highlight_name)
    highlights_out = cv2.VideoWriter(highlight_path, fourcc, fps, (W, H))

    tracking_out = None
    if write_tracking:
        tracking_out = cv2.VideoWriter(os.path.join(output_dir, "tracking.mp4"), fourcc, fps, (W, H))

    # Load/reuse single best.pt model
    model = _get_best_model(model_path)

    # Class names for best.pt
    class_names = ['Basketball', 'Basketball Hoop']

    # Shot detection state (ported from ShotDetector)
    ball_pos = []   # list of tuples: ((x, y), frame_idx, w, h, conf)
    hoop_pos = []   # list of tuples: ((x, y), frame_idx, w, h, conf)

    frame_count = 0
    makes = 0
    attempts = 0
    # Per-team breakdown using hoop side heuristic (right half -> Team A, left half -> Team B)
    team_a_makes = 0
    team_b_makes = 0

    up = False
    down = False
    up_frame = 0
    down_frame = 0

    # Score tracking with OCR
    ocr_team_a_score = None
    ocr_team_b_score = None
    last_team_a_score = None
    last_team_b_score = None
    score_check_interval = 30  # Check score every N frames
    score_change_frames = []   # Track frames where score changes

    # Clip durations (pre + post ~16s total to meet 15-20s requirement)
    clip_pre_seconds = 2   # was 8 — shorter lead-in
    clip_post_seconds = 4  # was 8 — shorter tail

    # Buffers for saving highlights around a make/miss
    pre_buffer = deque(maxlen=int(fps * clip_pre_seconds))  # up to ~8 seconds before
    post_frames = int(fps * clip_post_seconds)               # ~8 seconds after

    # Prediction kwargs for speed/accuracy balance
    half = (_DEVICE == 'cuda')
    pred_kwargs = dict(
        device=_DEVICE,
        imgsz=imgsz,
        conf=0.25,
        iou=0.6,
        classes=[0, 1],
        max_det=50,
        verbose=False,
        half=half,
    )

    # Optional warmup to amortize first-frame latency
    try:
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        _ = model.predict(dummy, **pred_kwargs)
        
        # Warmup OCR if enabled
        if use_ocr and (score_roi is not None or (team_a_roi is not None or team_b_roi is not None)):
            reader = get_ocr_reader()
            _ = reader.readtext(dummy, allowlist='0123456789')
    except Exception as e:
        print(f"Warmup error: {e}")
        pass

    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Ensure track_frame is always available and keep pre-buffer filled
            track_frame = frame.copy()
            pre_buffer.append(frame.copy())

            # If OCR is enabled but ROIs are missing, try to auto-detect once
            if use_ocr and score_roi is None and (team_a_roi is None or team_b_roi is None):
                try:
                    ta, tb, sr = auto_detect_scoreboard_rois(frame)
                    if ta is not None and tb is not None:
                        team_a_roi, team_b_roi, score_roi = ta, tb, sr
                        print(f"Auto-detected scoreboard ROIs: A={team_a_roi}, B={team_b_roi}, S={score_roi}")
                except Exception as e:
                    print(f"Auto ROI detection error: {e}")

            # OCR score detection (less frequent than object detection)
            if use_ocr and frame_count % score_check_interval == 0 and \
               (score_roi is not None or (team_a_roi is not None or team_b_roi is not None)):
                ocr_team_a_score, ocr_team_b_score, debug_frame = extract_scores_from_frame(
                    frame, score_roi, team_a_roi, team_b_roi, debug=write_tracking
                )
                
                # Detect score changes
                if (ocr_team_a_score is not None and last_team_a_score is not None and 
                    ocr_team_a_score > last_team_a_score):
                    score_change_frames.append((frame_count, 'A', ocr_team_a_score - last_team_a_score))
                    print(f"Team A score change: {last_team_a_score} -> {ocr_team_a_score}")
                
                if (ocr_team_b_score is not None and last_team_b_score is not None and 
                    ocr_team_b_score > last_team_b_score):
                    score_change_frames.append((frame_count, 'B', ocr_team_b_score - last_team_b_score))
                    print(f"Team B score change: {last_team_b_score} -> {ocr_team_b_score}")
                
                last_team_a_score = ocr_team_a_score
                last_team_b_score = ocr_team_b_score
                
                if write_tracking:
                    track_frame = debug_frame

            # Run detection
            res = model.predict(frame, **pred_kwargs)[0]

            # Parse detections similar to shot_detector.py
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    cls = int(box.cls[0]) if box.cls is not None else -1

                    if 0 <= cls < len(class_names):
                        current_class = class_names[cls]
                    else:
                        current_class = ""

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Ball points: high conf OR near hoop at slightly lower conf
                    if (
                        (conf > 0.30 or (in_hoop_region(center, hoop_pos) and conf > 0.15))
                        and current_class == 'Basketball'
                    ):
                        ball_pos.append((center, frame_count, w, h, conf))

                    # Hoop points: require higher confidence
                    if conf > 0.50 and current_class == 'Basketball Hoop':
                        hoop_pos.append((center, frame_count, w, h, conf))

            # Clean motion (utils)
            ball_pos = clean_ball_pos(ball_pos, frame_count)
            if len(hoop_pos) > 1:
                hoop_pos = clean_hoop_pos(hoop_pos)

            # Shot detection logic (utils)
            if len(hoop_pos) > 0 and len(ball_pos) > 0:
                if not up:
                    up = detect_up(ball_pos, hoop_pos)
                    if up:
                        up_frame = ball_pos[-1][1]

                if up and not down:
                    down = detect_down(ball_pos, hoop_pos)
                    if down:
                        down_frame = ball_pos[-1][1]

                # Every ~10 frames, check for a completed attempt
                if frame_count % 10 == 0:
                    if up and down and up_frame < down_frame:
                        attempts += 1
                        up = False
                        down = False

                        # Determine clip file info
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        event_id = f"{ts}_{frame_count:06d}"

                        # If it is a make, write highlight segment + per-event clip
                        if score(ball_pos, hoop_pos):
                            makes += 1

                            # Heuristic team assignment by hoop x position
                            if len(hoop_pos) > 0:
                                hoop_x = hoop_pos[-1][0][0]
                            elif len(ball_pos) > 0:
                                hoop_x = ball_pos[-1][0][0]
                            else:
                                hoop_x = W // 2
                            if hoop_x >= (W // 2):
                                team_a_makes += 1
                            else:
                                team_b_makes += 1

                            # Create event writer for make
                            make_clip_path = os.path.join(makes_dir, f"make_{event_id}.mp4")
                            make_out = cv2.VideoWriter(make_clip_path, fourcc, fps, (W, H))

                            # Write pre-buffered frames to both outputs
                            for f in pre_buffer:
                                make_out.write(f)
                                highlights_out.write(f)
                            # Write current frame
                            make_out.write(frame)
                            highlights_out.write(frame)

                            # Write a few seconds after to both
                            for _ in range(post_frames):
                                ret_post, post_frame = cap.read()
                                if not ret_post:
                                    break
                                highlights_out.write(post_frame)
                                make_out.write(post_frame)

                            make_out.release()
                            pre_buffer.clear()
                        else:
                            # Missed attempt: write per-event clip to miss folder and also keep in combined highlights
                            miss_clip_path = os.path.join(miss_dir, f"miss_{event_id}.mp4")
                            miss_out = cv2.VideoWriter(miss_clip_path, fourcc, fps, (W, H))

                            # Write pre-buffered frames to both outputs
                            for f in pre_buffer:
                                miss_out.write(f)
                                highlights_out.write(f)
                            # Write current frame
                            miss_out.write(frame)
                            highlights_out.write(frame)

                            # Write a few seconds after to both
                            for _ in range(post_frames):
                                ret_post, post_frame = cap.read()
                                if not ret_post:
                                    break
                                highlights_out.write(post_frame)
                                miss_out.write(post_frame)

                            miss_out.release()
                            pre_buffer.clear()

            # Add OCR info to tracking frame if available
            if write_tracking and (ocr_team_a_score is not None or ocr_team_b_score is not None):
                score_text = f"OCR: {ocr_team_a_score or '?'}-{ocr_team_b_score or '?'}"
                cv2.putText(track_frame, score_text, (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Add makes info
                makes_text = f"Makes: {makes} (A:{team_a_makes} B:{team_b_makes})"
                cv2.putText(track_frame, makes_text, (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            if tracking_out is not None:
                tracking_out.write(track_frame)

    cap.release()
    highlights_out.release()
    if tracking_out is not None:
        tracking_out.release()

    if makes == 0:
        # Remove empty highlights file
        try:
            os.remove(highlight_path)
        except Exception:
            pass
        return None, 0, 0, 0

    return highlight_name, makes, team_a_makes, team_b_makes
