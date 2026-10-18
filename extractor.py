import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import wave
import base64
import subprocess
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw
from faster_whisper import WhisperModel

MAX_SHORT_FORM_SEC = 180.0  # Strict 3-minute short-form ceiling
WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)


def _draw_ui_safe_zones(pil_img: Image.Image) -> Image.Image:
    """Draws translucent red TikTok/Reels/Shorts UI dead-zone overlays."""
    img = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size

    if h > w:  # Vertical 9:16
        draw.rectangle([int(w * 0.82), int(h * 0.40), w, int(h * 0.88)], fill=(255, 0, 0, 65), outline=(255, 50, 50, 180), width=2)
        draw.rectangle([0, int(h * 0.78), int(w * 0.82), h], fill=(255, 0, 0, 65), outline=(255, 50, 50, 180), width=2)
        draw.rectangle([0, 0, w, int(h * 0.08)], fill=(255, 0, 0, 45))
    else:  # Horizontal 16:9
        draw.rectangle([int(w * 0.85), int(h * 0.85), w, h], fill=(255, 0, 0, 65), outline=(255, 50, 50, 180), width=2)

    return Image.alpha_composite(img, overlay).convert("RGB")


def _pil_to_b64(pil_img: Image.Image, max_dim: int = 480, quality: int = 75) -> str:
    img = pil_img.copy()
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_12frame_storyboard_b64(story_frames: list) -> str:
    """
    Stitches 12 scene-aware frames into a 4-column x 3-row (960x780) Contact Sheet.
    Gives MiniCPM-V the complete narrative progression of a 1-3 minute video in ONE fast image payload.
    """
    cols, rows = 4, 3
    cell_w, cell_h = 240, 260
    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(18, 18, 18))
    draw = ImageDraw.Draw(grid)

    for idx, item in enumerate(story_frames[:12]):
        r = idx // cols
        c = idx % cols
        ox, oy = c * cell_w, r * cell_h

        thumb = item["pil"].copy()
        thumb.thumbnail((cell_w - 6, cell_h - 26), Image.Resampling.LANCZOS)
        tx = ox + (cell_w - thumb.width) // 2
        ty = oy + 22 + (cell_h - 22 - thumb.height) // 2
        grid.paste(thumb, (tx, ty))

        # Timestamp header bar on every panel
        draw.rectangle([ox, oy, ox + cell_w, oy + 20], fill=(0, 0, 0))
        ts = item["timestamp_sec"]
        mins, secs = int(ts // 60), int(ts % 60)
        label = f"#{idx + 1} [{mins}:{secs:02d}s]"
        draw.text((ox + 6, oy + 3), label, fill=(255, 220, 0))

    buf = BytesIO()
    grid.save(buf, format="JPEG", quality=76, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_visual_features(video_path: str) -> dict:
    """Scans up to 180s (3 mins), selects 12 scene-aware narrative frames + 4 UI safe-zone keyframes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_duration = max(total_frames / fps, 0.5)
    was_trimmed = raw_duration > MAX_SHORT_FORM_SEC
    duration_sec = min(raw_duration, MAX_SHORT_FORM_SEC)
    is_vertical = 1 if height > width else 0

    max_scan_frames = min(total_frames, int(duration_sec * fps))
    # Adaptive sampling: 4 fps for <60s clips, 2 fps for 60-180s clips (keeps CPU scan < 1 second)
    sample_hz = 4.0 if duration_sec <= 65.0 else 2.0
    step = max(int(fps / sample_hz), 1)
    hook_limit_frame = int(3.0 * fps)

    prev_gray = None
    prev_hist = None
    num_cuts = 0
    hook_motion_scores = []
    saturations = []
    brightnesses = []
    sampled_records = []  # Stores (frame_idx, timestamp_sec, change_score) for smart 12-frame selection

    for f_idx in range(0, max_scan_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        saturations.append(float(np.mean(hsv[:, :, 1])))
        brightnesses.append(float(np.mean(hsv[:, :, 2])))

        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        change_score = 0.0
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            change_score = max(0.0, 1.0 - corr)
            if corr < 0.62:
                num_cuts += 1
        prev_hist = hist

        motion_diff = 0.0
        if prev_gray is not None:
            motion_diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
            if f_idx <= hook_limit_frame:
                hook_motion_scores.append(motion_diff)
        prev_gray = gray

        # Combined visual salience score (scene cut + action motion)
        salience = (change_score * 50.0) + motion_diff
        sampled_records.append((f_idx, f_idx / fps, salience))

    cuts_per_min = round((num_cuts / max(duration_sec, 1.0)) * 60.0, 2)
    avg_shot_length = round(duration_sec / (num_cuts + 1), 2)

    # --- Select 12 Scene-Aware Frames across the full video timeline ---
    num_story_frames = 12
    bucket_len = duration_sec / num_story_frames
    selected_frame_indices = []

    for b in range(num_story_frames):
        b_start = b * bucket_len
        b_end = (b + 1) * bucket_len
        candidates = [r for r in sampled_records if b_start <= r[1] < b_end]
        if candidates:
            # Pick the frame in this time bucket with the highest scene change / action salience
            best = max(candidates, key=lambda x: x[2]) if b > 0 else candidates[0]
            selected_frame_indices.append((best[0], best[1]))
        else:
            ts = min(b_start + bucket_len * 0.5, duration_sec - 0.1)
            selected_frame_indices.append((min(int(ts * fps), max_scan_frames - 1), ts))

    story_frames_pil = []
    for f_idx, ts in selected_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            pil_raw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Draw safe zones on panels #1, #2, and #12 so VLM checks UI overlap AND full story
            pil_panel = _draw_ui_safe_zones(pil_raw) if len(story_frames_pil) in (0, 1, 11) else pil_raw
            story_frames_pil.append({"timestamp_sec": round(ts, 1), "pil": pil_panel})

    storyboard_b64 = _build_12frame_storyboard_b64(story_frames_pil)

    # Keep 4 UI Safe-Zone display frames for Tab 2 gallery
    ui_indices = [0, min(1, len(story_frames_pil) - 1), len(story_frames_pil) // 2, len(story_frames_pil) - 1]
    ui_keys = ["hook_1s", "hook_3s", "midpoint_50pct", "cta_90pct"]
    keyframes_b64 = {}
    for k_name, idx in zip(ui_keys, ui_indices):
        item = story_frames_pil[idx]
        pil_safe = _draw_ui_safe_zones(item["pil"])
        keyframes_b64[k_name] = {
            "timestamp_sec": item["timestamp_sec"],
            "b64_safezone": _pil_to_b64(pil_safe, max_dim=480),
        }

    cap.release()

    return {
        "duration_sec": round(duration_sec, 2),
        "raw_duration_sec": round(raw_duration, 2),
        "was_trimmed": was_trimmed,
        "is_vertical": is_vertical,
        "resolution": f"{width}x{height}",
        "cuts_per_min": cuts_per_min,
        "avg_shot_length_sec": avg_shot_length,
        "hook_motion_score": round(float(np.mean(hook_motion_scores)) if hook_motion_scores else 4.0, 2),
        "color_saturation": round(float(np.mean(saturations)) if saturations else 110.0, 2),
        "brightness_score": round(float(np.mean(brightnesses)) if brightnesses else 125.0, 2),
        "keyframes": keyframes_b64,
        "storyboard_b64": storyboard_b64,
    }


def extract_audio_and_transcript(video_path: str, duration_sec: float) -> dict:
    """Extracts up to 180s (3 mins) of audio and transcribes full narrative."""
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-t", str(int(MAX_SHORT_FORM_SEC)),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            temp_wav
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) < 1000:
            return {
                "transcript": "[No audible speech / Music or ambient audio only]",
                "segments": [],
                "wpm": 0.0,
                "hook_start_delay_sec": round(min(duration_sec, 5.0), 2),
                "dead_air_ratio": 0.5,
                "vocal_energy_variance": 0.0,
                "is_non_verbal": 1,
            }

        with wave.open(temp_wav, "rb") as wf:
            raw_bytes = wf.readframes(wf.getnframes())
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_np) > 1600:
            n_windows = len(audio_np) // 800
            windows = audio_np[: n_windows * 800].reshape(n_windows, 800)
            rms = np.sqrt(np.mean(windows ** 2, axis=1))
            energy_std = round(float(np.std(rms) * 1000.0), 2)
            acoustic_silence_ratio = round(float(np.mean(rms < 0.008)), 3)
        else:
            energy_std = 0.0
            acoustic_silence_ratio = 1.0

        segments_gen, _ = WHISPER_MODEL.transcribe(
            temp_wav,
            beam_size=1,
            best_of=1,
            vad_filter=True,
            word_timestamps=False
        )

        segments_list = []
        full_text_parts = []
        for seg in segments_gen:
            text_clean = seg.text.strip()
            if text_clean:
                segments_list.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": text_clean
                })
                full_text_parts.append(f"[{int(seg.start // 60)}:{int(seg.start % 60):02d}] {text_clean}")

        transcript_timed = " ".join(full_text_parts).strip()
        raw_words = " ".join([s["text"] for s in segments_list]).split()
        word_count = len(raw_words)
        wpm = round((word_count / max(duration_sec, 1.0)) * 60.0, 1)

        if word_count < 4:
            return {
                "transcript": "[Non-verbal / Music or visual-action clip — no spoken dialogue]",
                "segments": segments_list,
                "wpm": wpm,
                "hook_start_delay_sec": 0.5,
                "dead_air_ratio": acoustic_silence_ratio,
                "vocal_energy_variance": energy_std,
                "is_non_verbal": 1,
            }

        hook_start_delay = round(segments_list[0]["start"], 2)
        silence_total = hook_start_delay
        for i in range(1, len(segments_list)):
            gap = segments_list[i]["start"] - segments_list[i - 1]["end"]
            if gap > 0.45:
                silence_total += gap

        dead_air_ratio = round(min(silence_total / max(duration_sec, 1.0), 1.0), 3)

        return {
            "transcript": transcript_timed,
            "segments": segments_list,
            "wpm": wpm,
            "hook_start_delay_sec": hook_start_delay,
            "dead_air_ratio": dead_air_ratio,
            "vocal_energy_variance": energy_std,
            "is_non_verbal": 0,
        }
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def analyze_video_file(video_path: str) -> dict:
    """Master pipeline function for Short-Form Videos (up to 3 minutes)."""
    visual_data = extract_visual_features(video_path)
    audio_data = extract_audio_and_transcript(video_path, visual_data["duration_sec"])

    feature_vector = {
        "duration_sec": visual_data["duration_sec"],
        "is_vertical": visual_data["is_vertical"],
        "cuts_per_min": visual_data["cuts_per_min"],
        "avg_shot_length_sec": visual_data["avg_shot_length_sec"],
        "hook_motion_score": visual_data["hook_motion_score"],
        "color_saturation": visual_data["color_saturation"],
        "brightness_score": visual_data["brightness_score"],
        "wpm": audio_data["wpm"],
        "hook_start_delay_sec": audio_data["hook_start_delay_sec"],
        "dead_air_ratio": audio_data["dead_air_ratio"],
        "vocal_energy_variance": audio_data["vocal_energy_variance"],
        "is_non_verbal": audio_data["is_non_verbal"],
    }

    return {
        "features": feature_vector,
        "raw_duration_sec": visual_data["raw_duration_sec"],
        "was_trimmed": visual_data["was_trimmed"],
        "resolution": visual_data["resolution"],
        "transcript": audio_data["transcript"],
        "segments": audio_data["segments"],
        "keyframes": visual_data["keyframes"],
        "storyboard_b64": visual_data["storyboard_b64"],
    }