# Creator Co-Pilot AI

> **Built for AMD Developer Hackathon ACT III (October 2026)**  
> **Track:** Multimodal AI & Accelerated Computing with AMD Instinct & ROCm

---

## 📌 Executive Summary & Core Problem
Over **70 percent of short-form videos fail within the first 3 seconds** due to:
1. **Poor mobile typography contrast:** Captions blend into fluctuating background video lighting.
2. **UI safe-zone button collisions:** On-screen text, faces, and product CTAs get hidden behind native TikTok, Instagram Reels, and YouTube Shorts interface buttons, captions, and engagement icons.
3. **Weak narrative hooks and pacing drag:** Ineffective opening seconds and pacing bottlenecks fail to trigger recommendation algorithms.

**Creator Co-Pilot AI** is an intelligent pre-publish video advisor that performs automated multimodal audits, diagnostic machine learning scoring, and grounded AI rewrites to maximize audience retention and virality before hitting publish.

---

## ⚡ AMD Model Stack & Hardware Acceleration
Creator Co-Pilot AI leverages AMD Instinct cloud inference and the AMD Token Factory API gateway:

- **`MiniCPM-V-4.7-35B-A3B` (Multimodal Vision & Typography Audit):**
  - Ingests a high-density 12-frame scene-aware contact sheet in a single high-efficiency pass.
  - Audits typography legibility, contrast strokes, lighting, and overlays red safe-zone collision boundaries across critical timeline keyframes (Hook 0:01, 0:03, Midpoint 50%, and CTA/Ending 90%).
- **`DeepSeek-V4.1-Flash` (Narrative & Hook Rewriter):**
  - Grounded simultaneously in both the visual scene summary and the Whisper-transcribed audio timeline.
  - Generates calculated 0:00–0:03 hook upgrades, narrative pacing fixes, micro-copy punch-ups, and seamless loop suggestions tailored to target niches.
- **AMD Token Factory API Gateway:**
  - Base URL: `https://developer.amd.com.cn/radeon/api/v1`
  - Ultra-low-latency endpoint with 45-second resilient timeouts and automatic model fallback orchestration.

---

## 🏗️ System Architecture

1. **Multimodal Signal Extraction (`extractor.py`):**
   - **Scene-Aware Keyframe Selection:** Scans video histograms and frame-difference motion salience to synthesize a 12-frame narrative contact sheet (`#1` to `#12` with yellow timestamp headers).
   - **UI Safe-Zone Boundary Projection:** Automatically detects vertical 9:16 or horizontal 16:9 aspect ratios and renders translucent red overlays corresponding to TikTok/Reels UI dead-zones.
   - **Audio & Speech Analysis:** Powered by `faster-whisper` and `librosa` to compute speech pace (WPM), hook start latency, silence/dead-air ratios, and vocal energy variance.
2. **Diagnostic Reach & Benchmark Engine (`benchmark_model.py`):**
   - **Empirical Category Baselines:** Evaluates video metrics against top 5% benchmarks across four creator categories: *Tech & Tutorials*, *Short-Form Storytelling / Anime*, *Entertainment & Challenges*, and *Finance & Business*.
   - **XGBoost + SHAP Attribution:** Quantifies exact algorithmic bottlenecks and SHAP feature impact scores to highlight actionable retention fixes.
3. **AMD Multimodal Inference (`agents.py`):**
   - Dual-agent cooperative architecture executing visual critique and narrative hook synthesis.
4. **Interactive Dashboard (`app.py`):**
   - Real-time Streamlit web app with interactive Plotly SHAP visualizations, responsive storyboard galleries, and secure secret resolution.

---

## 🔒 Security & Docker Secret Architecture
Security and zero-trust credential hygiene are central to Creator Co-Pilot AI:

- **No Hardcoded Credentials:** API keys are never stored in source code.
- **No Image Layer Leaks:** Credentials are never baked into Docker images during build time. `.dockerignore` excludes `api_key.txt`, `.env`, and local virtual environments.
- **Version Control Protection:** `.gitignore` ensures `api_key.txt` and sensitive caches never get committed to GitHub.
- **Runtime Secret Mounting:** At container runtime, Docker Compose mounts `api_key.txt` as a standard Linux secret file at `/run/secrets/amd_api_key`.
- **Resilient Dual-Mode Resolution (`get_secret`):** The application inspects `/run/secrets/amd_api_key` first, gracefully falls back to the `AMD_API_KEY` environment variable, or allows configuration directly in the Streamlit UI sidebar without crashing on import.

---

## 🚀 Quickstart & Run Commands

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/) installed.
- Valid AMD Token Factory API key.

### Step 1: Provision Secret File
Create `api_key.txt` in the root project directory containing your AMD API key:
```bash
echo your_api_key_here > api_key.txt
```

### Step 2: Build & Launch with Docker Compose
```bash
docker compose up --build
```

The Streamlit dashboard will be live at:
👉 **`http://localhost:8501`**

---

### Local Development (Alternative)
If running locally without Docker:
```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Export API key
export AMD_API_KEY="your_api_key_here"  # On Windows: $env:AMD_API_KEY="your_api_key_here"

# 4. Launch app
streamlit run app.py
```

---

## 📊 Telemetry & Benchmark Categories
| Category | Viral Target Cuts/Min | Viral Target WPM | Hook Motion Index | Target Dead-Air |
| :--- | :---: | :---: | :---: | :---: |
| **Tech & Coding / Tutorials** | 11–18 cuts/min | 145–175 WPM | 8–18 | < 8% |
| **Short-Form Storytelling / Anime** | 14–22 cuts/min | 130–160 WPM | 14–25 | < 10% |
| **Entertainment & Challenges** | 18–30 cuts/min | 165–205 WPM | 18–35 | < 5% |
| **Finance & Business** | 12–19 cuts/min | 155–185 WPM | 8–16 | < 6% |

---

## 🏆 AMD Developer Hackathon ACT III
- **Team Praxis101**
- **Date:** October 2026
- **License:** Apache 2.0
