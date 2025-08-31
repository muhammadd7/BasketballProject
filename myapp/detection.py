# Updated to use single best.pt model and shot detection logic from utils.py
import cv2
import numpy as np
from collections import deque
import torch
from ultralytics import YOLO
import easyocr
import re
import time
import os

from utils import (
    get_device,
    score,
    detect_down,
    detect_up,
    in_hoop_region,
    clean_hoop_pos,
    clean_ball_pos,
)

# Initialize EasyOCR reader once (singleton pattern)
_OCR_READER = None

def get_ocr_reader():
    """Get or initialize the EasyOCR reader with optimal settings"""
    global _OCR_READER
    if _OCR_READER is None:
        device = get_device()
        gpu = device != 'cpu'
        # Use absolute path to the project's models/ocr directory
        base_dir = os.path.dirname(os.path.dirname(__file__))  # /Users/dev/Desktop/football_project
        ocr_model_dir = os.path.join(base_dir, 'models', 'ocr')
        _OCR_READER = easyocr.Reader(
            ['en'],
            gpu=gpu,
            model_storage_directory=ocr_model_dir,
            download_enabled=True,
            detector=True,
            recognizer=True,
            quantize=True,
            verbose=False
        )
    return _OCR_READER


def preprocess_scoreboard_roi(image, roi=None, debug=False):
    """Preprocess scoreboard region for better OCR accuracy
    
    Args:
        image: Full frame or ROI image
        roi: Region of interest (x, y, w, h) or None for full image
        debug: If True, return debug visualization
        
    Returns:
        Preprocessed image optimized for digit recognition
    """
    # Extract ROI if provided
    if roi is not None:
        x, y, w, h = roi
        if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
            # Fix out-of-bounds ROI
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
        roi_img = image[y:y+h, x:x+w]
    else:
        roi_img = image.copy()
    
    # Skip processing if ROI is too small
    if roi_img.shape[0] < 10 or roi_img.shape[1] < 10:
        return roi_img
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better digit separation
    # Use a larger block size for scoreboards with varying backgrounds
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 21, 10)
    
    # Noise removal
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilate to connect components of digits
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(opening, kernel, iterations=1)
    
    if debug:
        # Return visualization for debugging
        debug_img = np.hstack([gray, thresh, dilated])
        return cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
    
    # Return preprocessed binary image
    return dilated


def auto_detect_scoreboard_rois(frame):
    """
    Try common scoreboard locations to auto-detect ROI.
    Returns: (team_a_roi, team_b_roi, score_roi) or (None, None, None)
    """
    reader = get_ocr_reader()
    H, W = frame.shape[:2]

    candidates = [
        (int(W*0.35), int(H*0.02), int(W*0.30), int(H*0.10)),  # top-center
        (int(W*0.35), int(H*0.86), int(W*0.30), int(H*0.10)),  # bottom-center
        (int(W*0.02), int(H*0.02), int(W*0.30), int(H*0.10)),  # top-left
        (int(W*0.68), int(H*0.02), int(W*0.30), int(H*0.10)),  # top-right
        (int(W*0.02), int(H*0.86), int(W*0.30), int(H*0.10)),  # bottom-left
        (int(W*0.68), int(H*0.86), int(W*0.30), int(H*0.10)),  # bottom-right
    ]

    for (x, y, w, h) in candidates:
        roi_img = preprocess_scoreboard_roi(frame, (x, y, w, h))
        try:
            result = reader.readtext(
                roi_img,
                allowlist='0123456789-:',
                batch_size=1,
                detail=0,
                paragraph=True
            )
        except Exception:
            continue

        if not result:
            continue

        import re as _re
        text = ' '.join(result)
        scores = _re.findall(r'\d+', text)
        # Heuristic: need at least two numbers and reasonable values
        if len(scores) >= 2:
            try:
                a = int(scores[0])
                b = int(scores[1])
                # Basic sanity filter for scores
                if 0 <= a <= 200 and 0 <= b <= 200:
                    left_w = w // 2
                    team_a_roi = (x, y, left_w, h)
                    team_b_roi = (x + left_w, y, w - left_w, h)
                    score_roi = (x, y, w, h)
                    return team_a_roi, team_b_roi, score_roi
            except Exception:
                pass

    return None, None, None


def extract_scores_from_frame(frame, score_roi=None, team_a_roi=None, team_b_roi=None, debug=False):
    """Extract scores from a video frame using EasyOCR
    
    Args:
        frame: Video frame
        score_roi: Region containing both scores (x, y, w, h) or None
        team_a_roi: Region for team A score (x, y, w, h) or None
        team_b_roi: Region for team B score (x, y, w, h) or None
        debug: If True, return debug visualization
        
    Returns:
        (team_a_score, team_b_score, debug_frame)
    """
    reader = get_ocr_reader()
    debug_frame = frame.copy() if debug else None
    team_a_score = None
    team_b_score = None
    
    # Process team A ROI if provided
    if team_a_roi is not None:
        x, y, w, h = team_a_roi
        if debug:
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Preprocess ROI for better OCR
        roi_img = preprocess_scoreboard_roi(frame, team_a_roi)
        
        # Run OCR with optimized settings for digits
        result = reader.readtext(roi_img, 
                               allowlist='0123456789',
                               batch_size=1,
                               detail=0,
                               paragraph=False,
                               height_ths=0.6,
                               width_ths=0.6,
                               contrast_ths=0.1)
        
        # Extract score from OCR result
        if result:
            # Join all detected text and filter for digits only
            text = ''.join(result)
            digits = re.sub(r'\D', '', text)
            if digits:
                team_a_score = int(digits)
                if debug:
                    cv2.putText(debug_frame, f"A: {team_a_score}", (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Process team B ROI if provided
    if team_b_roi is not None:
        x, y, w, h = team_b_roi
        if debug:
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Preprocess ROI for better OCR
        roi_img = preprocess_scoreboard_roi(frame, team_b_roi)
        
        # Run OCR with optimized settings for digits
        result = reader.readtext(roi_img, 
                               allowlist='0123456789',
                               batch_size=1,
                               detail=0,
                               paragraph=False,
                               height_ths=0.6,
                               width_ths=0.6,
                               contrast_ths=0.1)
        
        # Extract score from OCR result
        if result:
            # Join all detected text and filter for digits only
            text = ''.join(result)
            digits = re.sub(r'\D', '', text)
            if digits:
                team_b_score = int(digits)
                if debug:
                    cv2.putText(debug_frame, f"B: {team_b_score}", (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # If we have a combined score ROI but no individual team ROIs
    if score_roi is not None and (team_a_roi is None or team_b_roi is None):
        x, y, w, h = score_roi
        if debug:
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Preprocess ROI for better OCR
        roi_img = preprocess_scoreboard_roi(frame, score_roi)
        
        # Run OCR with optimized settings for digits
        result = reader.readtext(roi_img, 
                               allowlist='0123456789-:',
                               batch_size=1,
                               detail=0,
                               paragraph=True)
        
        # Extract scores from OCR result
        if result:
            # Join all detected text
            text = ' '.join(result)
            if debug:
                cv2.putText(debug_frame, text, (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Try to extract two numbers separated by common delimiters
            scores = re.findall(r'\d+', text)
            if len(scores) >= 2:
                team_a_score = int(scores[0])
                team_b_score = int(scores[1])
    
    return team_a_score, team_b_score, debug_frame


def detect_goals_in_video(video_path: str,
                          model_path: str = "/Users/dev/Desktop/football_project/models/best.pt",
                          display: bool = False,
                          imgsz: int = 512,
                          frame_stride: int = 2,
                          score_roi: tuple = None,
                          team_a_roi: tuple = None,
                          team_b_roi: tuple = None,
                          use_ocr: bool = False) -> tuple:
    """
    Detects basketball goals using a single YOLO model (best.pt) that detects 'Basketball' and 'Basketball Hoop'.
    Uses shot detection logic (up/down regions + score()) adapted from shot_detector.py and utils.py.
    Optionally uses EasyOCR to detect scoreboard changes.

    Args:
        video_path (str): Path to the input video.
        model_path (str): Path to best.pt model with two classes.
        display (bool): If True, display the video with overlays.
        imgsz (int): Inference image size.
        frame_stride (int): Analyze every Nth frame for speed.
        score_roi (tuple): Region of interest for scoreboard (x, y, w, h).
        team_a_roi (tuple): Region of interest for team A score (x, y, w, h).
        team_b_roi (tuple): Region of interest for team B score (x, y, w, h).
        use_ocr (bool): If True, use OCR to detect scoreboard changes.

    Returns:
        tuple: (makes, team_a_score, team_b_score)
    """
    device = get_device()

    model = YOLO(model_path)
    try:
        model.fuse()
    except Exception:
        pass

    class_names = ['Basketball', 'Basketball Hoop']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return 0, None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    # Shot detection state
    ball_pos = []
    hoop_pos = []
    makes = 0
    attempts = 0
    up = False
    down = False
    up_frame = 0
    down_frame = 0

    # Score tracking with OCR
    team_a_score = None
    team_b_score = None
    last_team_a_score = None
    last_team_b_score = None
    score_check_interval = 30  # Check score every N frames
    score_change_frames = []  # Track frames where score changes

    # Debounce goal saving
    pre_buffer = deque(maxlen=int(fps * 6))

    # Prediction kwargs for speed/accuracy balance
    half = (device == 'cuda')
    pred_kwargs = dict(
        device=device,
        imgsz=imgsz,
        conf=0.25,
        iou=0.6,
        classes=[0, 1],
        max_det=50,
        verbose=False,
        half=half,
    )

    # Warmup
    try:
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        _ = model.predict(dummy, **pred_kwargs)
        
        # Warmup OCR if enabled
        if use_ocr and (score_roi is not None or (team_a_roi is not None and team_b_roi is not None)):
            reader = get_ocr_reader()
            _ = reader.readtext(dummy, allowlist='0123456789')
    except Exception as e:
        print(f"Warmup error: {e}")

    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_stride > 1 and (frame_idx % frame_stride != 0):
                pre_buffer.append(frame)
                continue

            pre_buffer.append(frame)
            
            # OCR score detection (less frequent than object detection)
            if use_ocr and frame_idx % score_check_interval == 0 and \
               (score_roi is not None or (team_a_roi is not None or team_b_roi is not None)):
                start_time = time.time()
                team_a_score, team_b_score, debug_frame = extract_scores_from_frame(
                    frame, score_roi, team_a_roi, team_b_roi, debug=display
                )
                
                # Detect score changes
                if (team_a_score is not None and last_team_a_score is not None and 
                    team_a_score > last_team_a_score):
                    score_change_frames.append((frame_idx, 'A', team_a_score - last_team_a_score))
                    print(f"Team A score change: {last_team_a_score} -> {team_a_score}")
                
                if (team_b_score is not None and last_team_b_score is not None and 
                    team_b_score > last_team_b_score):
                    score_change_frames.append((frame_idx, 'B', team_b_score - last_team_b_score))
                    print(f"Team B score change: {last_team_b_score} -> {team_b_score}")
                
                last_team_a_score = team_a_score
                last_team_b_score = team_b_score
                
                if display:
                    frame = debug_frame
                    ocr_time = (time.time() - start_time) * 1000
                    cv2.putText(frame, f"OCR: {ocr_time:.1f}ms", (20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Run object detection
            res = model.predict(frame, **pred_kwargs)[0]

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

                    if (conf > 0.30 or (in_hoop_region(center, hoop_pos) and conf > 0.15)) and current_class == 'Basketball':
                        ball_pos.append((center, frame_idx, w, h, conf))

                    if conf > 0.50 and current_class == 'Basketball Hoop':
                        hoop_pos.append((center, frame_idx, w, h, conf))

            ball_pos = clean_ball_pos(ball_pos, frame_idx)
            if len(hoop_pos) > 1:
                hoop_pos = clean_hoop_pos(hoop_pos)

            if len(hoop_pos) > 0 and len(ball_pos) > 0:
                if not up:
                    up = detect_up(ball_pos, hoop_pos)
                    if up:
                        up_frame = ball_pos[-1][1]

                if up and not down:
                    down = detect_down(ball_pos, hoop_pos)
                    if down:
                        down_frame = ball_pos[-1][1]

                if frame_idx % 10 == 0:
                    if up and down and up_frame < down_frame:
                        attempts += 1
                        up = False
                        down = False

                        if score(ball_pos, hoop_pos):
                            makes += 1

            if display:
                cv2.putText(frame, f"Makes: {makes}  Attempts: {attempts}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if team_a_score is not None and team_b_score is not None:
                    cv2.putText(frame, f"Score: {team_a_score}-{team_b_score}", (20, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Basketball Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    return makes, team_a_score, team_b_score
