import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_secret(secret_name: str, default: str = "") -> str:
    secret_path = f"/run/secrets/{secret_name}"
    if os.path.exists(secret_path):
        with open(secret_path, "r", encoding="utf-8") as f:
            val = f.read().strip()
            if val:
                return val
    return os.getenv(secret_name, default)


active_key = get_secret("amd_api_key", os.getenv("AMD_API_KEY", ""))

try:
    client = OpenAI(
        api_key=active_key if active_key else "EMPTY_KEY",
        base_url=os.getenv("AMD_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1"),
        timeout=45.0,
    )
except Exception:
    client = None


def run_visual_and_scene_critic(storyboard_b64: str, category: str) -> dict:
    """
    Sends the 12-frame chronological contact sheet (#1 to #12) to MiniCPM-V-4.7-35B-A3B.
    Captures the full beginning-to-end story arc of a 1-3 minute video in a single call.
    """
    current_key = get_secret("amd_api_key", os.getenv("AMD_API_KEY", ""))
    if not current_key or current_key == "EMPTY_KEY":
        return {
            "scene_summary": "12-frame visual storyboard extracted; manual scene inspection fallback active (no AMD API key provided).",
            "visual_audit": "- **Safe Zones:** Keep captions in the central safe zone.\n- **Typography:** Use high-contrast sans-serif text with a dark stroke.",
            "model_used": "Fallback (Missing API Key)",
        }

    global client
    if client is None or client.api_key != current_key:
        client = OpenAI(
            api_key=current_key,
            base_url=os.getenv("AMD_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1"),
            timeout=45.0,
        )

    prompt = f"""You are a senior short-form video editor for '{category}'. Look at this 12-frame chronological storyboard (#1 to #12 with timestamps in yellow). Panels #1, #2, and #12 also show red TikTok/Reels/Shorts UI dead-zone boxes.

Respond in this EXACT format (keep total response under 170 words, never cut off mid-sentence):

SCENE_SUMMARY: [3 chronological sentences tracing the full visual story across all 12 frames: (1) Hook/Setup in #1-#4, (2) Main progression/action in #5-#9, and (3) Climax/Ending in #10-#12, plus any visible text.]

VISUAL_AUDIT:
- **Typography & Contrast:** [1-2 sentences: evaluate existing on-screen text across the 12 frames or recommend exact font family, size, and contrast stroke for this lighting.]
- **UI Safe-Zone Check:** [1-2 sentences: check panels #1, #2, #12 for any text or subject overlap with the red right/bottom UI zones.]
- **Visual Story & Pacing Fix:** [1-2 sentences: identify which panel numbers (#1-#12) visually drag or repeat, and where to cut or zoom.]"""

    for model_name in ["MiniCPM-V-4.7-35B-A3B", "DeepSeek-V4-Flash-Vision-Exp"]:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{storyboard_b64}"}},
                    ],
                }],
                max_tokens=750,
                temperature=0.3,
            )
            raw_text = response.choices[0].message.content.strip()

            scene_summary = raw_text
            visual_audit = raw_text
            if "VISUAL_AUDIT:" in raw_text:
                parts = raw_text.split("VISUAL_AUDIT:", 1)
                scene_summary = parts[0].replace("SCENE_SUMMARY:", "").strip()
                visual_audit = parts[1].strip()

            return {
                "scene_summary": scene_summary,
                "visual_audit": visual_audit,
                "model_used": model_name,
            }
        except Exception:
            continue

    return {
        "scene_summary": "12-frame visual storyboard extracted; manual scene inspection fallback active.",
        "visual_audit": "- **Safe Zones:** Keep captions in the central safe zone.\n- **Typography:** Use high-contrast sans-serif text with a dark stroke.",
        "model_used": "Fallback",
    }


def run_storyline_advisor(
    transcript: str,
    scene_summary: str,
    features: dict,
    bottlenecks: list,
    category: str,
) -> str:
    """
    Uses DeepSeek-V4.1-Flash (fallback Qwen3.8-27B) grounded in BOTH the visual scene summary AND audio transcript.
    """
    current_key = get_secret("amd_api_key", os.getenv("AMD_API_KEY", ""))
    if not current_key or current_key == "EMPTY_KEY":
        return "Storyline analysis fallback: No active AMD API key found. Please mount via Docker secret or provide AMD_API_KEY."

    global client
    if client is None or client.api_key != current_key:
        client = OpenAI(
            api_key=current_key,
            base_url=os.getenv("AMD_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1"),
            timeout=45.0,
        )

    bottleneck_str = ", ".join(
        [f"{b['feature']}={b['value']} (target {b['target_benchmark']})" for b in bottlenecks]
    ) or "None"

    is_non_verbal = features.get("is_non_verbal", 0) == 1
    mode_instruction = (
        "NOTE: This video has NO spoken dialogue (it is a visual action/music clip). Do NOT invent a fake talking-head script. Instead, write on-screen text hook overlays, sound-effect/voiceover cues, and storyline pacing edits that match the exact visual scene."
        if is_non_verbal
        else "Ground every rewrite directly in the spoken transcript AND visual scene below. Do NOT invent unrelated topics."
    )

    prompt = f"""You are a viral Content Director for '{category}'.
{mode_instruction}

ACTUAL VIDEO CONTEXT:
- Visual Scene Progression (from VLM): {scene_summary}
- Audio Transcript (from Whisper): "{transcript}"
- Measured Metrics: Duration={features.get('duration_sec')}s | Cuts/min={features.get('cuts_per_min')} | WPM={features.get('wpm')} | Dead-Air={round(features.get('dead_air_ratio', 0)*100, 1)}%
- Top Algorithmic Bottlenecks: {bottleneck_str}

Write a calculated, high-impact advisory under 180 words total using these exact 4 bullets (max 2 sentences per bullet, no intro/outro fluff):
- **0:00–0:03 Hook Upgrade:** [Write the exact opening on-screen text + voiceover/sound hook tailored to this specific scene.]
- **Storyline & Pacing Edit:** [Exact edit to fix {bottleneck_str} and build tension from the setup to the 90% payoff.]
- **Naming / Micro-Copy Punch-Up:** [Suggest 2 punchier titles, character/subject labels, or callout phrases for this exact clip.]
- **Platform Fit & Loop Ending:** [Best platform (YT Shorts vs IG Reels), target viewer persona, and how to loop the final second back into 0:00.]"""

    last_err = "No models responded successfully."
    for model_name in ["DeepSeek-V4.1-Flash", "Qwen3.8-27B"]:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,  # Prevents cutoff; prompt enforces <180 words
                temperature=0.4,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_err = str(e)
            continue

    return f"Storyline analysis error: {last_err}"