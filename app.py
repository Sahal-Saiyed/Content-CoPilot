import os
import time
import hashlib
import streamlit as st
import pandas as pd
import plotly.express as px
from extractor import analyze_video_file
from benchmark_model import CategoryBenchmarkScorer
from agents import run_visual_and_scene_critic, run_storyline_advisor, get_secret

st.set_page_config(page_title="Creator Co-Pilot | AMD Hackathon ACT III", layout="wide")

# Detect active AMD Token Factory API key on startup
active_key = get_secret("amd_api_key", os.getenv("AMD_API_KEY", ""))


@st.cache_resource
def get_scorer():
    return CategoryBenchmarkScorer()


@st.cache_data(show_spinner=False)
def cached_extract_video(file_bytes: bytes) -> dict:
    """Caches extraction by file content so switching categories or tabs never re-runs FFmpeg/Whisper."""
    file_hash = hashlib.md5(file_bytes).hexdigest()[:10]
    temp_path = f"temp_{file_hash}.mp4"
    try:
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        return analyze_video_file(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


scorer = get_scorer()

st.title("🎬 Creator Co-Pilot AI")
st.caption("Pre-Publish Multimodal Video Advisor | Powered by AMD Instinct GPUs & ROCm Models")

with st.sidebar:
    st.header("⚙️ Project Settings")
    category = st.selectbox(
        "Content Niche / Category",
        [
            "Entertainment & Challenges",
            "Tech & Coding / Tutorials",
            "Short-Form Storytelling / Anime",
            "Finance & Business",
        ],
    )

    st.sidebar.header("Configuration")
    active_key = get_secret("amd_api_key", os.getenv("AMD_API_KEY", ""))
    api_key_input = st.sidebar.text_input(
        "AMD Token Factory API Key",
        value=active_key,
        type="password",
        help="Enter key if not mounted via Docker secrets",
    )
    if api_key_input:
        os.environ["AMD_API_KEY"] = api_key_input

    st.info(
        "⚡ **AMD Instinct Telemetry**\n"
        "- **Gateway:** AMD Token Factory\n"
        "- **Vision Agent:** `MiniCPM-V-4.7-35B-A3B`\n"
        "- **Reasoning Agent:** `DeepSeek-V4.1-Flash`\n"
        "- **ML Scorer:** `XGBoost + SHAP`"
    )

uploaded_file = st.file_uploader("Upload your draft video (.mp4, .mov)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    t0 = time.time()
    with st.spinner("1/2 Extracting visual & audio signals (fast single-pass)..."):
        analysis = cached_extract_video(uploaded_file.getvalue())
        if analysis.get("was_trimmed"):
            st.warning(
                f"⏱️ **Short-Form Scope Guardrail:** Your video is `{analysis['raw_duration_sec']}s` long. "
                "Creator Co-Pilot is optimized for **Short-Form Video (YouTube Shorts, IG Reels, TikTok — max 3:00 / 180s)**, "
                "so the first **180 seconds** were analyzed."
            )
        score_result = scorer.evaluate_video(analysis["features"], category=category)

    with st.spinner("2/2 Running AMD Multimodal Vision + Storyline Agents..."):
        vlm_result = run_visual_and_scene_critic(analysis["storyboard_b64"], category)
        storyline_advice = run_storyline_advisor(
            transcript=analysis["transcript"],
            scene_summary=vlm_result["scene_summary"],
            features=analysis["features"],
            bottlenecks=score_result["top_bottlenecks"],
            category=category,
        )
    elapsed = round(time.time() - t0, 1)

    # --- KPI ROW ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Virality Potential Score", f"{score_result['virality_score']} / 100")
    c2.metric("Detected Cuts / Min", f"{analysis['features']['cuts_per_min']}")
    c3.metric("Speech Pace (WPM)", "Non-Verbal / Action" if analysis["features"]["is_non_verbal"] else f"{analysis['features']['wpm']} WPM")
    c4.metric("Total Analysis Time", f"{elapsed}s")

    st.markdown(f"**👁️ Detected Visual Context:** *{vlm_result['scene_summary']}*")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "📊 Category Benchmark & Bottlenecks",
        "👁️ Visual & Safe-Zone Audit",
        "📝 Grounded Storyline & Hook Advisor",
    ])

    with tab1:
        st.subheader(f"SHAP Feature Impact vs. '{category}' Top 5% Benchmarks")
        feat_df = pd.DataFrame(score_result["feature_impacts"])
        fig = px.bar(
            feat_df,
            x="feature",
            y="impact",
            color="impact",
            hover_data=["value", "target_benchmark"],
            color_continuous_scale="RdYlGn",
            title=f"XGBoost + SHAP Attribution (Weights: {score_result['weights_source']})",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🚨 Top Algorithmic Bottlenecks to Fix")
        if score_result["top_bottlenecks"]:
            for b in score_result["top_bottlenecks"]:
                st.warning(
                    f"**{b['feature'].upper()}** — Measured: `{b['value']}` | Category Viral Target: `{b['target_benchmark']}` "
                    f"(SHAP Impact: `{b['impact']}`)"
                )
        else:
            st.success("All core pacing and retention metrics align with category top performers!")

    with tab2:
        st.subheader("12-Frame Chronological Storyboard & UI Safe-Zone Audit")
        with st.expander("🎞️ View Full 12-Frame Scene-Aware Contact Sheet Sent to MiniCPM-V", expanded=True):
            st.image(
                f"data:image/jpeg;base64,{analysis['storyboard_b64']}",
                caption="12 Scene-Aware Narrative Frames (#1 to #12) with Exact Timestamps",
                use_container_width=True,
            )

        keys = ["hook_1s", "hook_3s", "midpoint_50pct", "cta_90pct"]
        labels = ["Hook (0:01)", "Hook (0:03)", "Midpoint (50%)", "CTA / End (90%)"]
        cols = st.columns(4)
        for i, k in enumerate(keys):
            if k in analysis["keyframes"]:
                with cols[i]:
                    st.image(
                        f"data:image/jpeg;base64,{analysis['keyframes'][k]['b64_safezone']}",
                        caption=labels[i],
                        use_container_width=True,
                    )

        st.markdown(f"### 🤖 VLM Agent Recommendations (`{vlm_result['model_used']}`)")
        st.info(vlm_result["visual_audit"])

    with tab3:
        st.subheader("Combined Visual + Audio Narrative Optimization")
        col_a, col_b = st.columns(2)
        with col_a:
            st.text_area("Visual Scene Context (from MiniCPM-V)", value=vlm_result["scene_summary"], height=90)
        with col_b:
            st.text_area("Audio Transcript (from Faster-Whisper)", value=analysis["transcript"], height=90)

        st.markdown("### 🧠 Grounded Storyline & Hook Recommendations (`DeepSeek-V4.1-Flash`)")
        st.success(storyline_advice)