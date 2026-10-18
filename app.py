import os
import time
import hashlib
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from extractor import analyze_video_file
from benchmark_model import CategoryBenchmarkScorer
from agents import run_visual_and_scene_critic, run_storyline_advisor, get_secret

LOGO_FILE = "Content_Co-Pilot_Icon.png"
LOGO_PATH = LOGO_FILE if os.path.exists(LOGO_FILE) else None

st.set_page_config(
    page_title="Creator Co-Pilot | AMD Hackathon ACT III",
    page_icon=LOGO_PATH if LOGO_PATH else "🎬",
    layout="wide",
)

if LOGO_PATH:
    st.logo(LOGO_PATH)

# ── Custom CSS for polished dark UI ──────────────────────────────────────────
st.markdown("""
<style>
/* ── Metric cards ─────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1A1D23 0%, #22262E 100%);
    border: 1px solid #333840;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,.35);
}
div[data-testid="stMetric"] label {
    color: #9CA3AF !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #F9FAFB !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}

/* ── Tab styling ──────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* ── Sidebar branding ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1117 0%, #151921 100%);
}
section[data-testid="stSidebar"] hr {
    border-color: #2A2E35;
}

/* ── File uploader area ───────────────────────────────────────────── */
div[data-testid="stFileUploader"] {
    border: 2px dashed #ED1C24;
    border-radius: 12px;
    padding: 8px;
    transition: border-color 0.2s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #FF4D4D;
}

/* ── Expander styling ─────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid #333840;
    border-radius: 10px;
    background: #14171D;
}

/* ── Info/Warning/Success boxes ────────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* ── Divider ──────────────────────────────────────────────────────── */
hr {
    border-color: #2A2E35 !important;
}
</style>
""", unsafe_allow_html=True)

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

# ── Header ────────────────────────────────────────────────────────────────────
if LOGO_PATH:
    col_logo, col_title = st.columns([0.6, 10], vertical_alignment="center")
    with col_logo:
        st.image(LOGO_PATH, width=64)
    with col_title:
        st.markdown(
            '<h1 style="margin:0;padding:0;font-size:2.2rem;">Creator Co-Pilot AI</h1>',
            unsafe_allow_html=True,
        )
else:
    st.title("Creator Co-Pilot AI")

st.markdown(
    '<p style="color:#9CA3AF;margin-top:-8px;font-size:0.92rem;">'
    'Pre-Publish Multimodal Video Advisor &nbsp;·&nbsp; Powered by AMD Instinct GPUs &amp; ROCm Models'
    '</p>',
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, width=100)
    st.markdown("---")
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
    st.markdown("---")
    st.markdown(
        "⚡ **AMD Instinct Stack**\n\n"
        "| Component | Model |\n"
        "| :--- | :--- |\n"
        "| Gateway | AMD Token Factory |\n"
        "| Vision Agent | `MiniCPM-V-4.7-35B-A3B` |\n"
        "| Reasoning Agent | `DeepSeek-V4.1-Flash` |\n"
        "| ML Scorer | `XGBoost + SHAP` |"
    )
    st.markdown("---")
    st.caption("Built for AMD Developer Hackathon ACT III · Oct 2026")

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown("")
uploaded_file = st.file_uploader(
    "📁  Drop your draft video here (.mp4, .mov, .avi)",
    type=["mp4", "mov", "avi"],
)

if uploaded_file is None:
    # ── Hero empty-state ──────────────────────────────────────────────────────
    st.markdown("")
    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        st.markdown(
            """
            <div style="text-align:center;padding:50px 20px 40px 20px;border:1px solid #2A2E35;border-radius:16px;background:linear-gradient(135deg,#14171D 0%,#1A1F28 100%);">
                <h2 style="margin-bottom:8px;">🎯 Audit Your Next Viral Short</h2>
                <p style="color:#9CA3AF;font-size:1.05rem;max-width:520px;margin:0 auto 20px auto;">
                    Upload a short-form video and get instant AI-powered diagnostics on 
                    <strong>typography & contrast</strong>, <strong>UI safe-zone collisions</strong>, 
                    <strong>narrative hooks</strong>, and <strong>algorithmic reach scoring</strong>.
                </p>
                <p style="color:#6B7280;font-size:0.85rem;">
                    Supports MP4, MOV, AVI &nbsp;·&nbsp; Up to 3 minutes &nbsp;·&nbsp; Analysis in ~30s
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

# ── Analysis Pipeline ─────────────────────────────────────────────────────────
t0 = time.time()
with st.status("🔍 Analyzing your video...", expanded=True) as status_bar:
    st.write("**Step 1/3** — Extracting visual & audio signals...")
    analysis = cached_extract_video(uploaded_file.getvalue())
    if analysis.get("was_trimmed"):
        st.warning(
            f"⏱️ Video is `{analysis['raw_duration_sec']}s` — trimmed to first **180s** (short-form scope)."
        )

    st.write("**Step 2/3** — Scoring against category benchmarks...")
    score_result = scorer.evaluate_video(analysis["features"], category=category)

    st.write("**Step 3/3** — Running AMD Vision + Storyline Agents...")
    vlm_result = run_visual_and_scene_critic(analysis["storyboard_b64"], category)
    storyline_advice = run_storyline_advisor(
        transcript=analysis["transcript"],
        scene_summary=vlm_result["scene_summary"],
        features=analysis["features"],
        bottlenecks=score_result["top_bottlenecks"],
        category=category,
    )
    status_bar.update(label="✅ Analysis complete!", state="complete", expanded=False)

elapsed = round(time.time() - t0, 1)

# ── KPI Metrics Row ───────────────────────────────────────────────────────────
st.markdown("")
c1, c2, c3, c4 = st.columns(4, gap="medium")

score_val = score_result['virality_score']
score_color = "#22C55E" if score_val >= 75 else "#F59E0B" if score_val >= 50 else "#EF4444"

with c1:
    st.markdown(
        f"""<div style="text-align:center;">
        <div style="font-size:0.75rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.06em;">Virality Score</div>
        <div style="font-size:2.4rem;font-weight:800;color:{score_color};line-height:1.2;">{score_val}</div>
        <div style="font-size:0.75rem;color:#6B7280;">out of 100</div></div>""",
        unsafe_allow_html=True,
    )

c2.metric("Cuts / Min", f"{analysis['features']['cuts_per_min']}")
c3.metric(
    "Speech Pace",
    "Non-Verbal" if analysis["features"]["is_non_verbal"] else f"{analysis['features']['wpm']} WPM",
)
c4.metric("Analysis Time", f"{elapsed}s")

# ── Scene summary ─────────────────────────────────────────────────────────────
st.markdown("")
with st.expander("👁️ **Detected Visual Context** — click to expand", expanded=False):
    st.markdown(f"*{vlm_result['scene_summary']}*")

st.markdown("---")

# ── Tabbed Results ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Benchmark & Bottlenecks",
    "👁️  Visual & Safe-Zone Audit",
    "📝  Storyline & Hook Advisor",
])

# ── TAB 1: Benchmark ─────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"SHAP Feature Impact vs. '{category}' Top 5%")

    feat_df = pd.DataFrame(score_result["feature_impacts"])

    colors = ["#22C55E" if v >= 0 else "#EF4444" for v in feat_df["impact"]]

    fig = go.Figure(go.Bar(
        x=feat_df["feature"],
        y=feat_df["impact"],
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Impact: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"XGBoost + SHAP Attribution ({score_result['weights_source']})", font_size=14),
        xaxis_title=None,
        yaxis_title="SHAP Impact",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=45, b=10),
        height=380,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### 🚨 Top Algorithmic Bottlenecks")
    if score_result["top_bottlenecks"]:
        for b in score_result["top_bottlenecks"]:
            st.warning(
                f"**{b['feature'].upper()}** — Measured: `{b['value']}` → "
                f"Target: `{b['target_benchmark']}` &nbsp; (SHAP: `{b['impact']}`)"
            )
    else:
        st.success("🎉 All metrics align with category top performers!")

# ── TAB 2: Visual Audit ──────────────────────────────────────────────────────
with tab2:
    st.subheader("12-Frame Chronological Storyboard")
    st.image(
        f"data:image/jpeg;base64,{analysis['storyboard_b64']}",
        caption="Scene-Aware Contact Sheet (#1 → #12)  ·  Yellow = timestamps  ·  Red = UI dead-zones",
        width="stretch",
    )

    st.markdown("")
    st.markdown(f"#### 🤖 Vision Agent Recommendations &nbsp; `{vlm_result['model_used']}`")
    st.info(vlm_result["visual_audit"])

# ── TAB 3: Storyline Advisor ──────────────────────────────────────────────────
with tab3:
    st.subheader("Combined Visual + Audio Narrative Optimization")

    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        st.markdown("**🎥 Visual Scene Context** &nbsp; `MiniCPM-V`")
        st.text_area(
            "Visual Scene Context",
            value=vlm_result["scene_summary"],
            height=100,
            label_visibility="collapsed",
        )
    with col_b:
        st.markdown("**🎙️ Audio Transcript** &nbsp; `Faster-Whisper`")
        st.text_area(
            "Audio Transcript",
            value=analysis["transcript"],
            height=100,
            label_visibility="collapsed",
        )

    st.markdown("")
    st.markdown("#### 🧠 Grounded Storyline & Hook Recommendations &nbsp; `DeepSeek-V4.1-Flash`")
    st.markdown(storyline_advice)