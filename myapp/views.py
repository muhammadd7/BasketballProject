import os
import threading
import re
import datetime
import shutil
import time
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.http import JsonResponse
# from .goal_detector import detect_goals  # Removed to avoid ultralytics import in headless env
import subprocess
import glob
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()


# Resolve ffmpeg path cross-platform: allow env override, else use system PATH
ffmpeg_path = os.environ.get('FFMPEG_PATH') or shutil.which('ffmpeg') or 'ffmpeg'

video_processing_results = {}
detection_logs = {}

def _log(result_key, message):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {message}"
    print(line)
    buf = detection_logs.setdefault(result_key, [])
    buf.append(line)
    # keep last 1000 lines
    if len(buf) > 1000:
        detection_logs[result_key] = buf[-1000:]


def _parse_made_missed(text: str):
    import json
    try:
        data = json.loads(text)
        made = int(data.get('made'))
        missed = int(data.get('missed'))
        return made, missed
    except Exception:
        pass
    # Try regex parsing
    m = re.search(r'"?made"?\s*[:=]\s*(\d+)', text, re.IGNORECASE)
    n = re.search(r'"?miss(?:ed|es)?"?\s*[:=]\s*(\d+)', text, re.IGNORECASE)
    if m and n:
        return int(m.group(1)), int(n.group(1))
    nums = re.findall(r'\d+', text)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return 0, 0


def analyze_with_gemini(video_path: str, result_key: str):
    _log(result_key, "Gemini: initializing client")
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        _log(result_key, "Gemini: missing GEMINI_API_KEY/GOOGLE_API_KEY in environment")
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Upload video file to Gemini Files API
        mime = 'video/mp4'
        _log(result_key, f"Gemini: uploading video ({os.path.basename(video_path)})")
        gfile = genai.upload_file(path=video_path, mime_type=mime)

        # Poll until ACTIVE
        for i in range(60):
            gstate = genai.get_file(gfile.name)
            state = getattr(gstate, 'state', None)
            status = getattr(state, 'name', None) if state else None
            _log(result_key, f"Gemini: file status = {status}")
            if status == 'ACTIVE':
                gfile = gstate
                break
            time.sleep(2)
        else:
            _log(result_key, "Gemini: file did not become ACTIVE in time")
            return None

        model_name = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
        model = genai.GenerativeModel(model_name)
        prompt = """
            You are a basketball video analyst. Analyze the attached video and record EVERY unique SHOT ATTEMPT.

            Strict rules:
            - Only count live, on-camera attempts toward the basket.
            - Ignore replays, slow-motion, multiple angles, warmups, intros/outros, crowd shots, timeouts, and scoreboard-only segments.
            - Each free throw = one attempt.
            - If a shot goes in, mark it as "made".
            - If a shot misses, rims out, or is blocked, mark it as "missed".
            - Each attempt MUST include its timestamp (in seconds from the start of the video).
            - First, list every attempt. Then, totals MUST be derived ONLY by counting the list. Do not invent numbers.

            Output format:
            Return ONLY valid JSON, with no extra text, in this exact schema:

            {
            "attempts": [
                {"time": <float>, "result": "made|missed"},
                ...
            ],
            "totals": {
                "made": <int>,
                "missed": <int>
            }
            }
            """


        generation_config = {
            "temperature": 0,
            "response_mime_type": "application/json",
        }
        _log(result_key, f"Gemini: analyzing with model={model_name} (JSON-only)")
        resp = model.generate_content([gfile, prompt], generation_config=generation_config)
        text = getattr(resp, 'text', '') or ''
        _log(result_key, f"Gemini: raw response (truncated): {text[:200]}...")
        # Prefer strict JSON parsing to capture attempts and totals
        try:
            import json
            data = json.loads(text)
            attempts = []
            if isinstance(data, dict):
                # Support both {made, missed} and {attempts:[{time, result}], totals:{made, missed}}
                if 'attempts' in data and isinstance(data['attempts'], list):
                    for i, a in enumerate(data['attempts']):
                        try:
                            t = float(a.get('time'))
                            r = str(a.get('result', '')).strip().lower()
                            if r not in ('made', 'miss', 'missed'):
                                continue
                            r = 'made' if r == 'made' else 'missed'
                            attempts.append({'time': t, 'result': r})
                        except Exception:
                            continue
                totals = data.get('totals') if isinstance(data.get('totals'), dict) else data
                made = int(totals.get('made')) if 'made' in totals else None
                missed = int(totals.get('missed')) if 'missed' in totals else None
                if made is None or missed is None:
                    # Fallback if keys were absent
                    made, missed = _parse_made_missed(text)
            else:
                made, missed = _parse_made_missed(text)
                attempts = []
        except Exception:
            made, missed = _parse_made_missed(text)
            attempts = []
        _log(result_key, f"Gemini: parsed made={made}, missed={missed}, attempts_found={len(attempts)}")
        return {"made": int(made), "missed": int(missed), "attempts": attempts}
    except Exception as e:
        _log(result_key, f"Gemini: exception: {e}")
        return None


def process_video(video_path, result_key):
    output_dir = os.path.join(settings.BASE_DIR, 'myapp', 'static', 'uploaded_clips', result_key)
    os.makedirs(output_dir, exist_ok=True)

    # Derive a web path to the uploaded source (for fallback playback)
    source_rel_url = None
    try:
        static_root = os.path.join(settings.BASE_DIR, 'myapp', 'static')
        if video_path.startswith(static_root):
            sub = os.path.relpath(video_path, static_root).replace(os.sep, '/')
            source_rel_url = f"/static/{sub}"
    except Exception:
        source_rel_url = None

    # Store initial processing status for ETA reporting (heuristic 2 minutes)
    started_at = time.time()
    estimated_total_seconds = int(os.environ.get('ESTIMATED_TOTAL_SECONDS', '120'))
    video_processing_results[result_key] = {
        'status': 'processing',
        'started_at': started_at,
        'estimated_total_seconds': estimated_total_seconds,
        'source_url': source_rel_url,
    }
    _log(result_key, f"Started processing via Gemini. video_path={video_path}")

    # Analyze with Gemini for made/missed counts
    gm = analyze_with_gemini(video_path, result_key)

    if gm is None:
        # Finalize with failure
        video_processing_results[result_key] = {
            'status': 'done',
            'goal_count': 0,
            'team_a_makes': 0,
            'team_b_makes': 0,
            'ocr_team_a_score': None,
            'ocr_team_b_score': None,
            'clip_url': None,
            'makes_clip_url': None,
            'misses_clip_url': None,
            'miss_count': None,
        }
        _log(result_key, "Gemini analysis failed. Finalizing with zero results.")
        return

    made = int(gm.get('made', 0))
    missed = int(gm.get('missed', 0))
    attempts = gm.get('attempts') or []

    # If we have attempt timestamps, cut per-event clips and build compilations
    makes_url = None
    misses_url = None
    try:
        if attempts:
            _log(result_key, f"Starting clip extraction for {len(attempts)} attempts")
            makes_count, misses_count = cut_attempt_clips(video_path, attempts, output_dir, result_key)
            _log(result_key, f"Clip extraction done. makes={makes_count}, misses={misses_count}")
            # Concatenate
            makes_url = concat_subfolder(output_dir, 'make', 'compiled_makes.mp4', result_key)
            misses_url = concat_subfolder(output_dir, 'miss', 'compiled_misses.mp4', result_key)
            _log(result_key, f"Concat completed. makes_url={makes_url}, misses_url={misses_url}")
    except Exception as e:
        _log(result_key, f"Clip/concat error: {e}")

    # Finalize results based on Gemini output
    video_processing_results[result_key] = {
        'status': 'done',
        'goal_count': made,  # interpret 'made' as goals
        'team_a_makes': 0,   # team split not provided by Gemini prompt
        'team_b_makes': 0,
        'ocr_team_a_score': None,
        'ocr_team_b_score': None,
        'clip_url': None,            # maintain for legacy
        'makes_clip_url': makes_url,
        'misses_clip_url': misses_url,
        'miss_count': missed,        # additional field to expose misses
        'source_url': source_rel_url,
    }
    _log(result_key, f"Processing complete via Gemini. made={made}, missed={missed}")


@csrf_exempt
def chat_view(request):
    if request.method == 'POST' and 'video' in request.FILES:
        video = request.FILES['video']
        upload_dir = os.path.join(settings.BASE_DIR, 'myapp', 'static', 'uploaded')
        os.makedirs(upload_dir, exist_ok=True)

        file_key = f"{video.name}_{video.size}"
        video_path = os.path.join(upload_dir, f"{file_key}.mp4")

        with open(video_path, 'wb+') as destination:
            for chunk in video.chunks():
                destination.write(chunk)

        threading.Thread(target=process_video, args=(video_path, file_key)).start()

        return JsonResponse({'message': 'File uploaded. Processing started...', 'key': file_key})

    return render(request, 'chat.html', {'title': 'BasketBall Chatbot'})


def get_results(request):
    key = request.GET.get('key')
    print(f"Fetching result for key: {key}")
    result = video_processing_results.get(key)
    logs = detection_logs.get(key, [])[-200:]

    # If we have a processing record, return ETA and logs; also add timeout safeguard
    if result and result.get('status') == 'processing':
        est_total = result.get('estimated_total_seconds')
        started = result.get('started_at')
        remaining = None
        if est_total is not None and started is not None:
            elapsed = time.time() - started
            remaining = max(0, int(est_total - elapsed))
            # 5-minute grace beyond estimate, then auto-finalize to stop infinite polling
            if elapsed > (est_total + 300):
                # finalize with failure to stop polling
                video_processing_results[key] = {
                    'status': 'done',
                    'goal_count': 0,
                    'team_a_makes': 0,
                    'team_b_makes': 0,
                    'ocr_team_a_score': None,
                    'ocr_team_b_score': None,
                    'clip_url': None,
                }
                _log(key, f"Timeout exceeded (elapsed={int(elapsed)}s). Auto-finalizing as done with zero results.")
                return JsonResponse({'status': 'done', 'goal_count': 0, 'team_a_makes': 0, 'team_b_makes': 0, 'ocr_team_a_score': None, 'ocr_team_b_score': None, 'clip_url': None, 'logs': detection_logs.get(key, [])[-200:]})
        payload = {'status': 'processing'}
        if remaining is not None:
            payload['eta_seconds'] = remaining
        payload['logs'] = logs
        return JsonResponse(payload)

    if result:
        # Prepare compiled makes/misses early so we can decide whether to short-circuit
        makes_url = result.get('makes_clip_url')
        misses_url = result.get('misses_clip_url')

        raw_clip_url = result.get('clip_url')

        play_makes = None
        play_misses = None

        # Helper to remux one relative url into *_original.mp4 and return the new relative url
        def remux_rel(rel_url):
            if not rel_url:
                return None
            rel_trim = rel_url.lstrip('/')
            abs_path = os.path.join(settings.BASE_DIR, 'myapp', rel_trim.replace('/', os.sep))
            _log(key, f"Starting ffmpeg remux for playback: {abs_path}")
            ok = encode_with_ffmpeg(abs_path)
            if ok:
                return rel_url.rsplit('.mp4', 1)[0] + '_original.mp4'
            return None

        # Remux main highlight (if any)
        if raw_clip_url:
            raw_clip_url = raw_clip_url.lstrip('/')  # Remove leading slash
            abs_clip_path = os.path.join(settings.BASE_DIR, 'myapp', raw_clip_url.replace('/', os.sep))
            print(f"Absolute clip path: {abs_clip_path}")
            _log(key, f"Starting ffmpeg remux for playback: {abs_clip_path}")
            encoded_main_ok = encode_with_ffmpeg(abs_clip_path)
            encoded_rel_url = result['clip_url'].rsplit('.mp4', 1)[0] + '_original.mp4' if encoded_main_ok else None
        else:
            encoded_rel_url = None

        # Remux compiled makes/misses
        play_makes = remux_rel(makes_url)
        play_misses = remux_rel(misses_url)

        # If we still don't have a main playable URL, fall back to the uploaded source
        if not encoded_rel_url:
            encoded_rel_url = remux_rel(result.get('source_url'))

        # Build payload
        payload = {
            'status': 'done',
            'goal_count': result.get('goal_count', 0),
            'team_a_makes': result.get('team_a_makes', 0),
            'team_b_makes': result.get('team_b_makes', 0),
            'ocr_team_a_score': result.get('ocr_team_a_score'),
            'ocr_team_b_score': result.get('ocr_team_b_score'),
            'clip_url': result.get('clip_url'),
            'play_video': encoded_rel_url,
            'makes_clip_url': makes_url,
            'misses_clip_url': misses_url,
            'play_makes': play_makes,
            'play_misses': play_misses,
            'miss_count': result.get('miss_count'),
            'logs': detection_logs.get(key, [])[-200:],
        }
        return JsonResponse(payload)
    else:
        # No record yet: still initializing, return processing without ETA (include any early logs)
        return JsonResponse({'status': 'processing', 'logs': logs})


def encode_with_ffmpeg(input_path):
    print("FFmpeg encoding started")
    input_path = os.path.normpath(input_path)
    output_path = os.path.normpath(input_path.rsplit('.mp4', 1)[0] + '_original.mp4')
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        command = [
            ffmpeg_path,
            '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-movflags', '+faststart',
            output_path
        ]
        subprocess.run(command, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("❌ FFmpeg error:", e)
        return False


def search_videos(request):
    query = request.GET.get('query', '').lower()
    results = []
    
    # Get all uploaded videos
    upload_dir = os.path.join(settings.BASE_DIR, 'myapp', 'static', 'uploaded')
    video_files = glob.glob(os.path.join(upload_dir, '*.mp4'))
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        # Extract the original filename (before the key)
        original_name = re.sub(r'_\d+\.mp4$', '', filename)
        
        # Check if the video has been processed
        file_key = filename.rsplit('.mp4', 1)[0]
        result = video_processing_results.get(file_key)
        
        # Get file stats
        stats = os.stat(video_path)
        modified_time = datetime.datetime.fromtimestamp(stats.st_mtime)
        date_str = modified_time.strftime('%Y-%m-%d %H:%M')
        
        # Only include if it matches the search query
        if query in original_name.lower():
            video_info = {
                'key': file_key,
                'name': original_name,
                'date': date_str,
                'size': stats.st_size,
                'path': video_path,
                'processed': result is not None
            }
            
            # Add goal count if available
            if result:
                video_info['goal_count'] = result.get('goal_count', 0)
                video_info['team_a_makes'] = result.get('team_a_makes', 0)
                video_info['team_b_makes'] = result.get('team_b_makes', 0)
                video_info['ocr_team_a_score'] = result.get('ocr_team_a_score')
                video_info['ocr_team_b_score'] = result.get('ocr_team_b_score')
            
            results.append(video_info)
    
    # Sort by date (newest first)
    results.sort(key=lambda x: x['date'], reverse=True)
    
    return JsonResponse({'results': results})


# --- New helpers for cutting and concatenation ---
def _ffprobe_duration_seconds(path):
    try:
        cmd = [ffmpeg_path.replace('ffmpeg', 'ffprobe') if ffmpeg_path.endswith('ffmpeg') else 'ffprobe',
               '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8', errors='ignore').strip()
        return float(out)
    except Exception:
        return None


def cut_attempt_clips(video_path, attempts, output_dir, result_key, pre_roll=2.0, post_roll=3.0):
    """
    For each attempt, cut a short clip around the timestamp into make/ and miss/ subfolders.
    Returns (makes_count, misses_count)
    """
    # Prepare folders and cleanup old mp4s
    make_dir = os.path.join(output_dir, 'make')
    miss_dir = os.path.join(output_dir, 'miss')
    os.makedirs(make_dir, exist_ok=True)
    os.makedirs(miss_dir, exist_ok=True)
    for p in glob.glob(os.path.join(make_dir, '*.mp4')) + glob.glob(os.path.join(miss_dir, '*.mp4')):
        try:
            os.remove(p)
        except Exception:
            pass

    duration = _ffprobe_duration_seconds(video_path)
    if duration is not None:
        _log(result_key, f"Video duration: {duration:.2f}s")

    make_idx = 1
    miss_idx = 1
    for i, att in enumerate(attempts):
        try:
            t = float(att.get('time', 0))
            res = str(att.get('result', 'missed')).lower()
            res = 'made' if res == 'made' else 'missed'
            start = max(0.0, t - pre_roll)
            end = t + post_roll
            if duration is not None:
                end = min(duration, end)
            seg_dur = max(0.2, end - start)
            out_dir = make_dir if res == 'made' else miss_dir
            idx = make_idx if res == 'made' else miss_idx
            out_name = f"{('make' if res=='made' else 'miss')}_{idx:03d}.mp4"
            out_path = os.path.join(out_dir, out_name)
            cmd = [
                ffmpeg_path, '-y',
                '-ss', f"{start:.2f}", '-t', f"{seg_dur:.2f}",
                '-i', video_path,
                '-c:v', 'libx264', '-preset', 'veryfast',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                out_path
            ]
            _log(result_key, f"ffmpeg cut: t={t:.2f}s res={res} -> {out_path}")
            subprocess.run(cmd, check=True)
            if res == 'made':
                make_idx += 1
            else:
                miss_idx += 1
        except subprocess.CalledProcessError as e:
            _log(result_key, f"ffmpeg cut failed at attempt {i}: {e}")
        except Exception as e:
            _log(result_key, f"cut exception at attempt {i}: {e}")

    return (make_idx - 1, miss_idx - 1)


def concat_subfolder(base_dir, subfolder, output_filename, result_key):
    folder = os.path.join(base_dir, subfolder)
    files = sorted(glob.glob(os.path.join(folder, '*.mp4')))
    if not files:
        _log(result_key, f"No clips found in {folder} to concatenate")
        return None
    list_path = os.path.join(base_dir, f"{subfolder}_list.txt")
    try:
        with open(list_path, 'w') as f:
            for p in files:
                f.write(f"file '{p}'\n")
        out_path = os.path.join(base_dir, output_filename)
        # First try stream copy
        cmd_copy = [ffmpeg_path, '-y', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', out_path]
        _log(result_key, f"ffmpeg concat copy -> {out_path}")
        try:
            subprocess.run(cmd_copy, check=True)
        except subprocess.CalledProcessError:
            # Fallback to re-encode
            cmd_enc = [ffmpeg_path, '-y', '-f', 'concat', '-safe', '0', '-i', list_path, '-c:v', 'libx264', '-c:a', 'aac', '-movflags', '+faststart', out_path]
            _log(result_key, f"ffmpeg concat encode -> {out_path}")
            subprocess.run(cmd_enc, check=True)
        rel = f"/static/uploaded_clips/{os.path.basename(base_dir)}/{output_filename}"
        return rel
    except Exception as e:
        _log(result_key, f"concat error ({subfolder}): {e}")
        return None
