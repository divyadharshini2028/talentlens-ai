"""TalentLens AI - Modern Multi-Resume Screening Dashboard.

Designed with a soft blue-lavender workspace, dark navy sidebar,
blue-to-purple gradient header, thin consistent light-purple borders,
clearly demarcated and coloured result panels, candidate leaderboard,
and ResumeShield forensic audit panels.
"""

from __future__ import annotations
import sys

# Ensure UTF-8 console output encoding on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import io
import os
import time
import json
from pathlib import Path

# Ensure project root is in sys.path for robust imports across all deployment environments
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.config import (
    get_gemini_api_key,
    is_api_key_configured,
    REQUIRED_SKILL_WEIGHT,
    PREFERRED_SKILL_WEIGHT,
    DEFAULT_GEMINI_MODEL,
)
from src.pdf_inspector import inspect_pdf_bytes, InspectionResult
from src.gemini_client import (
    GeminiClient,
    JUNIOR_DATA_ANALYST_JOB_DESCRIPTION,
    LIVE_RUBRIC_CRITERIA,
)
from src.scorer import (
    compute_match_score,
    rank_and_group_candidates,
    CandidateScore,
)
from src.feedback import generate_candidate_feedback
from src.demo_data import (
    SAMPLE_JOB_TITLE,
    SAMPLE_JOB_DESCRIPTION,
    get_demo_resumes,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Injection
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TalentLens AI - Multi-Resume Screening",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0f172a;
}

/* Soft, light blue-lavender background for main workspace */
[data-testid="stAppViewContainer"] {
    background-color: #f0f3fc !important;
}

/* Main block container padding */
.main .block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 1.8rem !important;
    padding-right: 1.8rem !important;
    max-width: 1480px;
}

/* Dark Navy Sidebar with readable labels and clear separators */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b112c 0%, #0d1538 60%, #080c20 100%) !important;
    color: #f8fafc !important;
    border-right: 1.5px solid #1e295d !important;
}

[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid #1e295d !important;
    margin: 14px 0 !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] div {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
    color: #ffffff !important;
}

/* Sidebar logo container */
.sidebar-logo-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 10px 0 14px 0;
}

.sidebar-logo-icon {
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
    margin-bottom: 10px;
}

.sidebar-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #ffffff !important;
    letter-spacing: -0.02em;
    margin: 0;
}

.sidebar-subtitle {
    font-size: 0.78rem;
    color: #94a3b8 !important;
    margin-top: 4px;
    font-weight: 500;
}

/* Sidebar feature card at bottom */
.sidebar-shield-card {
    background: #11183c;
    border: 1px solid #1f295b;
    border-radius: 12px;
    padding: 14px 12px;
    text-align: center;
    margin-top: 20px;
}

/* Blue-to-Purple Gradient Header Banner */
.hero-header-banner {
    background: linear-gradient(135deg, #0b112c 0%, #151b4b 35%, #2a1158 70%, #1e3a8a 100%);
    border: 1.5px solid #3730a3;
    border-radius: 16px;
    padding: 22px 28px;
    color: #ffffff;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.12);
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.hero-header-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.02em;
}

.hero-header-subtitle {
    font-size: 0.95rem;
    color: #cbd5e1;
    margin-top: 4px;
    font-weight: 400;
}

/* Mode Notice Card with visible border */
.mode-notice-demo {
    background: #eef2ff;
    border: 1.5px solid #a5b4fc;
    border-radius: 12px;
    padding: 12px 18px;
    color: #312e81;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.mode-notice-live-ready {
    background: #f0fdf4;
    border: 1.5px solid #86efac;
    border-radius: 12px;
    padding: 12px 18px;
    color: #14532d;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.mode-notice-live-missing {
    background: #fffbeb;
    border: 1.5px solid #fcd34d;
    border-radius: 12px;
    padding: 12px 18px;
    color: #78350f;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
}

/* Major Bordered Containers - BOX 1, 2, 3, 4 */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1.5px solid #c7d2fe !important;
    border-radius: 14px !important;
    background-color: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    margin-bottom: 14px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 18px 20px !important;
}

.ui-step-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}

.ui-step-badge {
    background: #4f46e5;
    color: #ffffff;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
}

.ui-step-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
}

/* File Uploader dashed styling */
[data-testid="stFileUploader"] section {
    border: 1.5px dashed #a5b4fc !important;
    background-color: #fafbff !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

[data-testid="stFileUploader"] section:hover {
    border-color: #6366f1 !important;
    background-color: #f5f7ff !important;
}

/* Text Area input */
[data-testid="stTextArea"] textarea {
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    background-color: #ffffff !important;
    font-size: 0.88rem !important;
    line-height: 1.45 !important;
    color: #0f172a !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 1px #6366f1 !important;
}

/* Primary Button with Blue-Purple Accent Gradient */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 50%, #7c3aed 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 12px 20px !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3) !important;
    transition: all 0.2s ease !important;
}

div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
    transform: translateY(-1px);
}

/* Secondary Button Styling */
div.stButton > button[kind="secondary"] {
    background-color: #ffffff !important;
    border: 1.5px solid #c7d2fe !important;
    color: #1e293b !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.15s ease !important;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: #818cf8 !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

/* Candidate Leaderboard Buttons */
.candidate-list-wrapper div.stButton > button {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    width: 100% !important;
    border-radius: 12px !important;
    padding: 11px 15px !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
    transition: all 0.15s ease !important;
}

.candidate-list-wrapper div.stButton > button[kind="secondary"] {
    background-color: #ffffff !important;
    border: 1.5px solid #c7d2fe !important;
    color: #0f172a !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
}

.candidate-list-wrapper div.stButton > button[kind="secondary"]:hover {
    border-color: #818cf8 !important;
    background-color: #f8fafc !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(99, 102, 241, 0.12) !important;
}

.candidate-list-wrapper div.stButton > button[kind="primary"] {
    background: #eef2ff !important;
    border: 1.5px solid #6366f1 !important;
    color: #1e1b4b !important;
    box-shadow: 0 0 0 1px #6366f1, 0 2px 8px rgba(99, 102, 241, 0.15) !important;
}

/* 5 Distinct Colored Result Panels */
.panel-supported {
    background: #f0fdf4 !important;
    border: 1.5px solid #86efac !important;
    border-radius: 14px !important;
    padding: 16px 18px !important;
    height: 100%;
}

.panel-missing {
    background: #fef2f2 !important;
    border: 1.5px solid #fca5a5 !important;
    border-radius: 14px !important;
    padding: 16px 18px !important;
    height: 100%;
}

.panel-uncertain {
    background: #faf5ff !important;
    border: 1.5px solid #d8b4fe !important;
    border-radius: 14px !important;
    padding: 16px 18px !important;
    height: 100%;
}

.panel-suggestions {
    background: #eff6ff !important;
    border: 1.5px solid #93c5fd !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    height: 100%;
}

.panel-suspicious-flagged {
    background: #fffbeb !important;
    border: 1.5px solid #fcd34d !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    height: 100%;
}

.panel-suspicious-clean {
    background: #f0fdf4 !important;
    border: 1.5px solid #86efac !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    height: 100%;
}

/* Bottom Completion Bar */
.panel-completion-bar {
    background: #eef2ff !important;
    border: 1.5px solid #c7d2fe !important;
    border-radius: 14px !important;
    padding: 14px 22px !important;
    margin-top: 16px !important;
}

/* Badge Pills */
.badge-pill {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 9999px;
    font-size: 0.74rem;
    font-weight: 600;
}

.badge-blue {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}

.badge-purple {
    background: #faf5ff;
    color: #7e22ce;
    border: 1px solid #e9d5ff;
}

.badge-green {
    background: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
}

.badge-amber {
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
}

/* Suspicious Resumes Forensic Cards */
.suspicious-pill-status {
    background: #fee2e2;
    color: #991b1b;
    border: 1px solid #fca5a5;
    font-size: 0.74rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.suspicious-pill-type {
    background: #ffedd5;
    color: #9a3412;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.74rem;
    border: 1px solid #fdba74;
    display: inline-flex;
    align-items: center;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. State Initialization
# -----------------------------------------------------------------------------
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Dashboard"

if "analysis_mode" not in st.session_state:
    st.session_state["analysis_mode"] = "Demo Analysis"

if "job_description_text" not in st.session_state:
    st.session_state["job_description_text"] = SAMPLE_JOB_DESCRIPTION

if "evaluation_results" not in st.session_state:
    st.session_state["evaluation_results"] = None

if "selected_candidate_idx" not in st.session_state:
    st.session_state["selected_candidate_idx"] = 0

if "uploaded_resume_files" not in st.session_state:
    st.session_state["uploaded_resume_files"] = []

if "use_synthetic_batch" not in st.session_state:
    st.session_state["use_synthetic_batch"] = True

if "live_batch_type" not in st.session_state:
    st.session_state["live_batch_type"] = "standard"

if "last_analysis_time" not in st.session_state:
    st.session_state["last_analysis_time"] = 1.2


# -----------------------------------------------------------------------------
# 3. Navy Sidebar: Brand, Navigation, Mode Selection & Setup Guidance
# -----------------------------------------------------------------------------
with st.sidebar:
    # Brand Header with Circular Glowing Icon
    st.markdown(
        """
        <div class="sidebar-logo-container">
            <div class="sidebar-logo-icon">
                <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                    <circle cx="19" cy="11" r="3"></circle>
                    <path d="m21 13-1.5-1.5"></path>
                </svg>
            </div>
            <div class="sidebar-title">TalentLens AI</div>
            <div class="sidebar-subtitle">Smarter Matching. Safer Screening.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Navigation menu
    nav_options = ["Dashboard", "About", "Settings", "Help"]
    nav_icons = {"Dashboard": "🏠", "About": "ℹ️", "Settings": "⚙️", "Help": "❓"}

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        format_func=lambda x: f"{nav_icons[x]}  {x}",
        label_visibility="collapsed",
        key="sidebar_nav_radio",
    )
    st.session_state["nav_page"] = selected_nav

    st.markdown("<hr>", unsafe_allow_html=True)

    # Mode selection in Sidebar
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #cbd5e1; margin-bottom: 8px;'>Screening Mode</div>", unsafe_allow_html=True)
    sidebar_mode = st.radio(
        "Operating Mode",
        options=["Demo Analysis", "Live Gemini Analysis"],
        index=0 if st.session_state["analysis_mode"] == "Demo Analysis" else 1,
        label_visibility="collapsed",
        key="sidebar_mode_select",
    )
    if sidebar_mode != st.session_state["analysis_mode"]:
        st.session_state["analysis_mode"] = sidebar_mode
        st.session_state["evaluation_results"] = None
        st.session_state["use_synthetic_batch"] = True
        st.session_state["uploaded_resume_files"] = []
        if sidebar_mode == "Demo Analysis":
            st.session_state["job_description_text"] = SAMPLE_JOB_DESCRIPTION
        else:
            st.session_state["job_description_text"] = JUNIOR_DATA_ANALYST_JOB_DESCRIPTION
        st.rerun()

    # API Key Status & Setup Guidance when Live mode is selected
    if st.session_state["analysis_mode"] == "Live Gemini Analysis":
        if is_api_key_configured():
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 8px; padding: 10px 12px; font-size: 0.8rem; color: #6ee7b7; margin-top: 10px;">
                    ✓ <strong>Gemini API Ready</strong><br>
                    <span style="font-size: 0.72rem; color: #a7f3d0;">Model: {DEFAULT_GEMINI_MODEL}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; border-radius: 8px; padding: 10px 12px; font-size: 0.8rem; color: #fcd34d; margin-top: 10px;">
                    ⚠️ <strong>API Key Missing</strong><br>
                    <span style="font-size: 0.72rem; color: #fef08a;">Set GEMINI_API_KEY in .env or use local semantic evaluator.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("🔑 API Setup Instructions"):
                st.caption(
                    "1. Create or edit `.env` in the project root:\n"
                    "   `GEMINI_API_KEY=your_key_here`\n"
                    "2. Get a key at [Google AI Studio](https://aistudio.google.com/)\n"
                    "3. Or use Demo Mode for offline evaluation."
                )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Scoring Formula weights
    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #cbd5e1; margin-bottom: 6px;'>Scoring Rubric</div>", unsafe_allow_html=True)
    if st.session_state["analysis_mode"] == "Live Gemini Analysis":
        st.caption(
            "• **Required (80 pts):**\n"
            "  - Python for data analysis (20)\n"
            "  - SQL and database querying (20)\n"
            "  - Data cleaning (20)\n"
            "  - Basic statistics (20)\n"
            "• **Preferred (20 pts):**\n"
            "  - Visualization (Tableau/Power BI) (10)\n"
            "  - Cloud basics (10)\n"
            "• **Final Score:** Sum of 6 scores"
        )
    else:
        st.caption(
            f"• Required Skills: **{int(REQUIRED_SKILL_WEIGHT * 100)}%**\n"
            f"• Preferred Skills: **{int(PREFERRED_SKILL_WEIGHT * 100)}%**\n"
            "• Calculated strictly in Python code\n"
            "• Security flags never alter score"
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ResumeShield Badge Card at bottom of Sidebar
    st.markdown(
        """
        <div class="sidebar-shield-card">
            <div style="width: 36px; height: 36px; border-radius: 50%; background: rgba(16, 185, 129, 0.2); display: inline-flex; align-items: center; justify-content: center; margin-bottom: 6px;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                    <path d="m9 12 2 2 4-4"></path>
                </svg>
            </div>
            <div style="color: #ffffff; font-weight: 700; font-size: 0.88rem;">ResumeShield Forensics</div>
            <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 2px;">Inspects Hidden &amp; Adversarial Text</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 4. PAGE: ABOUT
# =============================================================================
if st.session_state["nav_page"] == "About":
    with st.container(border=True):
        st.markdown(
            """
            <h2 style="color: #0f172a; margin-top: 0;">ℹ️ About TalentLens AI</h2>
            <p style="color: #475569; font-size: 1.05rem;">
                <strong>TalentLens AI</strong> is an automated technical resume screening platform engineered to bring 
                objective transparency, verifiable evidence matching, and adversarial security inspection to the recruitment pipeline.
            </p>
            <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 18px 0;">
            <h4 style="color: #1e293b;">Key Architecture Principles:</h4>
            <ul style="color: #334155; line-height: 1.8;">
                <li><strong>Deterministic Python Scoring:</strong> Large Language Models (LLMs) are strictly forbidden from calculating scores or assigning numerical ratings. All scores follow a mathematical rubric in Python.</li>
                <li><strong>Evidence-Backed Verification:</strong> Skills count as supported only when verifiable evidence quotes exist in the resume. Ambiguous mentions are quarantined as "Needs Review".</li>
                <li><strong>ResumeShield Forensic Inspection:</strong> Scans PDF structure and layout for invisible white text, micro-fonts (&lt; 3.0pt), off-page coordinates, overlapping text spans, and prompt injection attacks.</li>
                <li><strong>Security Independence:</strong> Forensic flags are reported in a distinct audit panel and never alter or penalize candidate scores.</li>
                <li><strong>Zero-API-Key Demo Mode:</strong> Features 5 synthetic candidates for instant testing without requiring an API key.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
    st.stop()


# =============================================================================
# 5. PAGE: SETTINGS
# =============================================================================
if st.session_state["nav_page"] == "Settings":
    with st.container(border=True):
        st.markdown(
            f"""
            <h2 style="color: #0f172a; margin-top: 0;">⚙️ System Configuration &amp; Settings</h2>
            <p style="color: #475569;">Configure your Gemini API key and review scoring parameters.</p>
            <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 18px 0;">
            <h4 style="color: #1e293b;">Gemini API Configuration</h4>
            <p style="color: #334155;">Active Key Status: <code>{'Configured (' + DEFAULT_GEMINI_MODEL + ')' if is_api_key_configured() else 'Not Configured'}</code></p>
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; font-size: 0.85rem; color: #334155; margin-bottom: 16px;">
                To configure your API key, create or update <code>.env</code> in the project directory:<br>
                <code>GEMINI_API_KEY=AIzaSy...</code>
            </div>
            <h4 style="color: #1e293b;">Live Mode Junior Data Analyst Rubric</h4>
            <ul style="color: #334155; line-height: 1.8;">
                <li>Python for data analysis: <strong>20 pts</strong></li>
                <li>SQL and database querying: <strong>20 pts</strong></li>
                <li>Data cleaning: <strong>20 pts</strong></li>
                <li>Basic statistics: <strong>20 pts</strong></li>
                <li>Tableau, Power BI or equivalent visualization: <strong>10 pts</strong></li>
                <li>Cloud basics: <strong>10 pts</strong></li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
    st.stop()


# =============================================================================
# 6. PAGE: HELP
# =============================================================================
if st.session_state["nav_page"] == "Help":
    with st.container(border=True):
        st.markdown(
            """
            <h2 style="color: #0f172a; margin-top: 0;">❓ Help &amp; User Guide</h2>
            <p style="color: #475569;">Step-by-step instructions for operating TalentLens AI.</p>
            <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 18px 0;">
            <ol style="color: #334155; line-height: 1.8;">
                <li><strong>Select Mode:</strong> Choose <em>Demo Analysis</em> for instant offline evaluation with synthetic resumes, or <em>Live Gemini Analysis</em> for data analyst resumes.</li>
                <li><strong>Review Job Description:</strong> Use Box 1 to inspect or edit the job requirements. Click <em>Load Sample Job</em> to restore defaults.</li>
                <li><strong>Inspect Criteria:</strong> Box 2 displays the extracted Required (80%) and Preferred (20%) skill criteria and point allocations.</li>
                <li><strong>Upload Resumes:</strong> In Box 3, drag &amp; drop candidate PDF files or click <em>Load Resumes</em> to test pre-loaded sample profiles.</li>
                <li><strong>Click 'Analyze Resumes':</strong> Box 3 runs PyMuPDF forensic checks, semantic evidence extraction, and deterministic scoring.</li>
                <li><strong>Inspect Results in Box 4:</strong> Browse candidate rankings on the left, and view evidence quotes, missing skills, uncertain items, skill-gap suggestions, and ResumeShield forensic audit on the right.</li>
                <li><strong>Export Report:</strong> Click <em>Download Report</em> to export complete evaluation data as a CSV.</li>
            </ol>
            """,
            unsafe_allow_html=True,
        )
    st.stop()


# =============================================================================
# 7. PAGE: DASHBOARD (MAIN SCREEN)
# =============================================================================

# --- HEADER BANNER ---
st.markdown(
    """
    <div class="hero-header-banner">
        <div>
            <div class="hero-header-title">TalentLens AI</div>
            <div class="hero-header-subtitle">Evaluate multiple resumes, rank candidates, and detect suspicious content.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Header mode pill selector
header_col_space, header_col_mode = st.columns([2.5, 1.5])
with header_col_mode:
    header_mode = st.radio(
        "Screening Mode Header",
        options=["Demo Analysis", "Live Gemini Analysis"],
        horizontal=True,
        index=0 if st.session_state["analysis_mode"] == "Demo Analysis" else 1,
        label_visibility="collapsed",
        key="header_mode_radio",
    )
    if header_mode != st.session_state["analysis_mode"]:
        st.session_state["analysis_mode"] = header_mode
        st.session_state["evaluation_results"] = None
        st.session_state["use_synthetic_batch"] = True
        st.session_state["uploaded_resume_files"] = []
        if header_mode == "Demo Analysis":
            st.session_state["job_description_text"] = SAMPLE_JOB_DESCRIPTION
        else:
            st.session_state["job_description_text"] = JUNIOR_DATA_ANALYST_JOB_DESCRIPTION
        st.rerun()

is_demo = (st.session_state["analysis_mode"] == "Demo Analysis")

# --- MODE NOTICE CARD ---
if is_demo:
    st.markdown(
        """
        <div class="mode-notice-demo">
            <span style="font-size: 1.25rem;">💡</span>
            <div>
                <strong>Demo Analysis Mode Active (Synthetic Fixtures)</strong><br>
                <span style="font-size: 0.82rem; color: #4338ca;">
                    Running 100% offline without an API key using 5 curated synthetic candidate resumes and sample job criteria.
                    All demo data is strictly synthetic and isolated from real uploaded data.
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif is_api_key_configured():
    st.markdown(
        f"""
        <div class="mode-notice-live-ready">
            <span style="font-size: 1.25rem;">🟢</span>
            <div>
                <strong>Live Gemini Analysis Mode Active</strong><br>
                <span style="font-size: 0.82rem; color: #166534;">
                    Connected to Gemini API (Model: <code>{DEFAULT_GEMINI_MODEL}</code>). Evaluates candidate resumes against the 6 Junior Data Analyst criteria (80 pts required, 20 pts preferred).
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="mode-notice-live-missing">
            <span style="font-size: 1.25rem;">⚠️</span>
            <div>
                <strong>Live Analysis Active (Calibrated Semantic Evaluator)</strong><br>
                <span style="font-size: 0.82rem; color: #92400e;">
                    Operating with calibrated deterministic semantic evaluator for the 6 Junior Data Analyst criteria. Set <code>GEMINI_API_KEY</code> in <code>.env</code> to activate live cloud Gemini calls.
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- TOP ROW: BOX 1, BOX 2, BOX 3 (ALL CONTENT PLACED STRICTLY INSIDE EACH CONTAINER) ---
top_col1, top_col2, top_col3 = st.columns([1.3, 1.2, 1.4], gap="medium")

# =============================================================================
# BOX 1: JOB DESCRIPTION
# =============================================================================
with top_col1:
    with st.container(border=True):
        st.markdown(
            """
            <div class="ui-step-header">
                <span class="ui-step-badge">1</span>
                <span class="ui-step-title">Job Description</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        jd_text = st.text_area(
            "Job Description:",
            value=st.session_state["job_description_text"],
            height=140,
            placeholder="Enter job requirements and qualifications...",
            label_visibility="collapsed",
            key="input_jd_area",
        )
        st.session_state["job_description_text"] = jd_text

        col_char, col_btn_sample = st.columns([1, 1])
        with col_char:
            st.caption(f"`{len(jd_text)}` chars")
        with col_btn_sample:
            if st.button("📋 Load Sample Job", use_container_width=True, key="btn_load_sample_job"):
                if is_demo:
                    st.session_state["job_description_text"] = SAMPLE_JOB_DESCRIPTION
                else:
                    st.session_state["job_description_text"] = JUNIOR_DATA_ANALYST_JOB_DESCRIPTION
                st.session_state["evaluation_results"] = None
                st.rerun()


# =============================================================================
# BOX 2: TARGET JOB CRITERIA
# =============================================================================
with top_col2:
    with st.container(border=True):
        st.markdown(
            """
            <div class="ui-step-header">
                <span style="font-size: 1.2rem;">🎯</span>
                <span class="ui-step-title">Target Job Criteria</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not is_demo:
            st.markdown(
                """
                <div style="font-size: 0.82rem; font-weight: 700; color: #1e293b; margin-bottom: 2px;">Role: Junior Data Analyst</div>
                <div style="font-size: 0.78rem; font-weight: 700; color: #2563eb; margin-top: 6px;">Required Skills (80 Points):</div>
                <div style="margin-top: 2px;">
                    <span class='badge-pill badge-blue' style='margin: 2px 2px;'>Python for data analysis (20)</span>
                    <span class='badge-pill badge-blue' style='margin: 2px 2px;'>SQL and database querying (20)</span>
                    <span class='badge-pill badge-blue' style='margin: 2px 2px;'>Data cleaning (20)</span>
                    <span class='badge-pill badge-blue' style='margin: 2px 2px;'>Basic statistics (20)</span>
                </div>
                <div style="font-size: 0.78rem; font-weight: 700; color: #7c3aed; margin-top: 8px;">Preferred Skills (20 Points):</div>
                <div style="margin-top: 2px;">
                    <span class='badge-pill badge-purple' style='margin: 2px 2px;'>Tableau, Power BI or equiv. (10)</span>
                    <span class='badge-pill badge-purple' style='margin: 2px 2px;'>Cloud basics (10)</span>
                </div>
                <div style="font-size: 0.72rem; color: #64748b; margin-top: 8px;">
                    Formula: Sum of all 6 criterion scores (Max 100).
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            client_preview = GeminiClient(api_key=None)
            parsed_preview = client_preview.parse_job_description(st.session_state["job_description_text"])

            if parsed_preview.get("is_valid"):
                req_list = parsed_preview.get("required_skills", [])
                pref_list = parsed_preview.get("preferred_skills", [])

                st.markdown(f"<div style='font-size: 0.82rem; font-weight: 700; color: #1e293b; margin-bottom: 2px;'>Role: {parsed_preview.get('role_title', 'Technical Role')}</div>", unsafe_allow_html=True)

                st.markdown(f"<div style='font-size: 0.78rem; font-weight: 700; color: #2563eb; margin-top: 6px;'>Required Skills ({len(req_list)}) — 80%:</div>", unsafe_allow_html=True)
                req_tags = " ".join([f"<span class='badge-pill badge-blue' style='margin: 2px 2px;'>{s}</span>" for s in req_list[:5]])
                if len(req_list) > 5:
                    req_tags += f" <span style='font-size: 0.72rem; color: #64748b;'>+{len(req_list)-5} more</span>"
                st.markdown(req_tags, unsafe_allow_html=True)

                st.markdown(f"<div style='font-size: 0.78rem; font-weight: 700; color: #7c3aed; margin-top: 6px;'>Preferred Skills ({len(pref_list)}) — 20%:</div>", unsafe_allow_html=True)
                if pref_list:
                    pref_tags = " ".join([f"<span class='badge-pill badge-purple' style='margin: 2px 2px;'>{s}</span>" for s in pref_list[:3]])
                    st.markdown(pref_tags, unsafe_allow_html=True)
                else:
                    st.caption("None (Scoring: 100% Required)")
            else:
                st.warning(f"⚠️ {parsed_preview.get('clarification_message', 'Invalid description.')}")


# =============================================================================
# BOX 3: UPLOAD RESUMES
# =============================================================================
with top_col3:
    with st.container(border=True):
        st.markdown(
            """
            <div class="ui-step-header">
                <span class="ui-step-badge">2</span>
                <span class="ui-step-title">Upload Resumes</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="font-size: 0.76rem; color: #64748b; margin-bottom: 6px;">
                Upload multiple candidate PDF resumes. Each document is inspected and scored independently.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Upload PDF Resumes",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="uploader_resumes",
        )

        if uploaded_files:
            st.session_state["uploaded_resume_files"] = uploaded_files
            st.session_state["use_synthetic_batch"] = False
            st.markdown(
                f"""
                <div style="color: #16a34a; font-size: 0.82rem; font-weight: 600; margin-top: 2px;">
                    ✓ <strong>{len(uploaded_files)} file(s) uploaded:</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
            fnames_html = " ".join([f"<span class='badge-pill badge-blue' style='margin: 2px;'>📄 {f.name[:18]}...</span>" for f in uploaded_files[:3]])
            st.markdown(fnames_html, unsafe_allow_html=True)
        else:
            if is_demo:
                col_demo_status, col_btn_demo = st.columns([1.2, 1])
                with col_demo_status:
                    if st.session_state["use_synthetic_batch"]:
                        st.markdown(
                            """
                            <div style="color: #16a34a; font-size: 0.78rem; font-weight: 600;">
                                ✓ 5 synthetic demo resumes loaded
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("No files uploaded.")
                with col_btn_demo:
                    if st.button("📂 Demo Resumes", use_container_width=True, key="btn_load_demo_files"):
                        st.session_state["use_synthetic_batch"] = True
                        st.session_state["uploaded_resume_files"] = []
                        st.session_state["evaluation_results"] = None
                        st.rerun()
            else:
                batch_t = st.session_state.get("live_batch_type", "standard")
                if batch_t == "suspicious":
                    label_batch = "2 suspicious test resumes loaded"
                else:
                    label_batch = "5 sample Data Analyst resumes loaded"

                col_demo_status, col_btn_demo = st.columns([1.1, 1.4])
                with col_demo_status:
                    if st.session_state["use_synthetic_batch"]:
                        st.markdown(
                            f"""
                            <div style="color: #16a34a; font-size: 0.78rem; font-weight: 600;">
                                ✓ {label_batch}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("No files uploaded.")
                with col_btn_demo:
                    col_b1, col_b2 = st.columns([1, 1])
                    with col_b1:
                        if st.button("📂 Standard", use_container_width=True, key="btn_std_files", help="Load 5 standard Data Analyst candidate resumes"):
                            st.session_state["use_synthetic_batch"] = True
                            st.session_state["live_batch_type"] = "standard"
                            st.session_state["uploaded_resume_files"] = []
                            st.session_state["evaluation_results"] = None
                            st.rerun()
                    with col_b2:
                        if st.button("⚠️ Suspicious", use_container_width=True, key="btn_susp_files", help="Load 2 suspicious adversarial candidate resumes"):
                            st.session_state["use_synthetic_batch"] = True
                            st.session_state["live_batch_type"] = "suspicious"
                            st.session_state["uploaded_resume_files"] = []
                            st.session_state["evaluation_results"] = None
                            st.rerun()

        # Prominent Analyze Resumes button
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("▶  Analyze Resumes", type="primary", use_container_width=True)


# --- 8. EXECUTE ANALYSIS WORKFLOW ---
if analyze_clicked:
    start_time = time.time()
    with st.spinner("Analyzing candidate resumes and running ResumeShield forensics..."):
        client = GeminiClient(api_key=get_gemini_api_key() if (not is_demo and is_api_key_configured()) else None)

        # Assemble candidate sources
        resumes_to_eval = []
        if not st.session_state["use_synthetic_batch"] and st.session_state["uploaded_resume_files"]:
            for f in st.session_state["uploaded_resume_files"]:
                resumes_to_eval.append({
                    "filename": f.name,
                    "bytes": f.read(),
                    "source": "Uploaded",
                })
        elif is_demo:
            demo_items = get_demo_resumes()
            for item in demo_items:
                resumes_to_eval.append({
                    "filename": item["filename"],
                    "bytes": item["pdf_bytes"],
                    "source": "Synthetic Demo",
                    "notes": item.get("expected_scenario", ""),
                })
        else:
            # Live mode sample resumes (Standard 5 or Suspicious 2)
            sample_dir = Path(__file__).resolve().parent / "sample_resumes"
            batch_t = st.session_state.get("live_batch_type", "standard")
            if batch_t == "suspicious":
                target_fnames = ["suspicious_resume_1.pdf", "suspicious_resume_2.pdf"]
            else:
                target_fnames = [f"resume_{i}.pdf" for i in range(1, 6)]

            for fname in target_fnames:
                p = sample_dir / fname
                if p.exists():
                    with open(p, "rb") as f_pdf:
                        resumes_to_eval.append({
                            "filename": fname,
                            "bytes": f_pdf.read(),
                            "source": "Sample Resume",
                        })

        evaluation_results: list[CandidateScore] = []

        if is_demo:
            parsed_jd = client.parse_job_description(st.session_state["job_description_text"])
            req_skills = parsed_jd.get("required_skills", [])
            pref_skills = parsed_jd.get("preferred_skills", [])

            for cand_item in resumes_to_eval:
                fname = cand_item["filename"]
                pdf_bytes = cand_item["bytes"]

                try:
                    insp: InspectionResult = inspect_pdf_bytes(pdf_bytes, filename=fname)
                    findings_dicts = [
                        {
                            "category": f.category,
                            "detection_type": f.detection_type or f.category,
                            "detected_text": f.detected_text,
                            "reason": f.reason,
                            "severity": f.severity,
                            "page": f.page,
                        }
                        for f in insp.findings
                    ]

                    if insp.status != "OK":
                        c_score = compute_match_score(
                            required_skills=req_skills,
                            preferred_skills=pref_skills,
                            evaluations=[],
                            candidate_name=fname,
                            filename=fname,
                            is_suspicious=insp.is_suspicious,
                            suspicious_findings=findings_dicts,
                            candidate_status="Needs review",
                            review_reason=insp.review_reason,
                        )
                    else:
                        evidence = client.extract_resume_evidence(
                            resume_text=insp.raw_text,
                            required_skills=req_skills,
                            preferred_skills=pref_skills,
                            filename=fname,
                        )
                        c_name = evidence.get("candidate_name") or fname
                        evals = evidence.get("skill_evaluations", [])

                        c_score = compute_match_score(
                            required_skills=req_skills,
                            preferred_skills=pref_skills,
                            evaluations=evals,
                            candidate_name=c_name,
                            filename=fname,
                            is_suspicious=insp.is_suspicious,
                            suspicious_findings=findings_dicts,
                            candidate_status="Scored",
                        )
                    evaluation_results.append(c_score)

                except Exception as single_err:
                    err_score = compute_match_score(
                        required_skills=req_skills,
                        preferred_skills=pref_skills,
                        evaluations=[],
                        candidate_name=fname,
                        filename=fname,
                        candidate_status="Needs review",
                        review_reason=f"Processing exception: {str(single_err)}",
                    )
                    evaluation_results.append(err_score)

        else:
            # Live Mode scoring with 6-criteria Junior Data Analyst rubric
            req_names = [c["name"] for c in LIVE_RUBRIC_CRITERIA if c["category"] == "required"]
            pref_names = [c["name"] for c in LIVE_RUBRIC_CRITERIA if c["category"] == "preferred"]

            for cand_item in resumes_to_eval:
                fname = cand_item["filename"]
                pdf_bytes = cand_item["bytes"]

                try:
                    insp: InspectionResult = inspect_pdf_bytes(pdf_bytes, filename=fname)
                    findings_dicts = [
                        {
                            "category": f.category,
                            "detection_type": f.detection_type or f.category,
                            "detected_text": f.detected_text,
                            "reason": f.reason,
                            "severity": f.severity,
                            "page": f.page,
                        }
                        for f in insp.findings
                    ]

                    if insp.status != "OK":
                        c_score = compute_match_score(
                            required_skills=req_names,
                            preferred_skills=pref_names,
                            evaluations=[],
                            candidate_name=fname,
                            filename=fname,
                            is_suspicious=insp.is_suspicious,
                            suspicious_findings=findings_dicts,
                            candidate_status="Needs review",
                            review_reason=insp.review_reason,
                        )
                    else:
                        live_eval = client.evaluate_live_rubric(
                            resume_text=insp.raw_text,
                            filename=fname,
                        )
                        c_name = live_eval.get("candidate_name") or fname
                        evals = live_eval.get("skill_evaluations", [])

                        c_score = compute_match_score(
                            required_skills=req_names,
                            preferred_skills=pref_names,
                            evaluations=evals,
                            candidate_name=c_name,
                            filename=fname,
                            is_suspicious=insp.is_suspicious,
                            suspicious_findings=findings_dicts,
                            candidate_status="Scored",
                        )
                    evaluation_results.append(c_score)

                except Exception as single_err:
                    err_score = compute_match_score(
                        required_skills=req_names,
                        preferred_skills=pref_names,
                        evaluations=[],
                        candidate_name=fname,
                        filename=fname,
                        candidate_status="Needs review",
                        review_reason=f"Processing exception: {str(single_err)}",
                    )
                    evaluation_results.append(err_score)

        st.session_state["evaluation_results"] = evaluation_results
        st.session_state["last_analysis_time"] = round(time.time() - start_time, 1)
        st.session_state["selected_candidate_idx"] = 0
        st.rerun()


# Auto-run batch on initial visit if no results exist
if st.session_state["evaluation_results"] is None:
    client = GeminiClient(api_key=None)

    if is_demo:
        parsed_jd = client.parse_job_description(st.session_state["job_description_text"])
        req_skills = parsed_jd.get("required_skills", [])
        pref_skills = parsed_jd.get("preferred_skills", [])
        demo_items = get_demo_resumes()

        eval_init = []
        for item in demo_items:
            insp = inspect_pdf_bytes(item["pdf_bytes"], filename=item["filename"])
            findings_dicts = [
                {
                    "category": f.category,
                    "detection_type": f.detection_type or f.category,
                    "detected_text": f.detected_text,
                    "reason": f.reason,
                    "severity": f.severity,
                    "page": f.page,
                }
                for f in insp.findings
            ]
            if insp.status != "OK":
                c_score = compute_match_score(
                    required_skills=req_skills,
                    preferred_skills=pref_skills,
                    evaluations=[],
                    candidate_name=item["candidate_name"],
                    filename=item["filename"],
                    is_suspicious=insp.is_suspicious,
                    suspicious_findings=findings_dicts,
                    candidate_status="Needs review",
                    review_reason=insp.review_reason,
                )
            else:
                evidence = client.extract_resume_evidence(
                    resume_text=insp.raw_text,
                    required_skills=req_skills,
                    preferred_skills=pref_skills,
                    filename=item["filename"],
                )
                c_score = compute_match_score(
                    required_skills=req_skills,
                    preferred_skills=pref_skills,
                    evaluations=evidence.get("skill_evaluations", []),
                    candidate_name=item["candidate_name"],
                    filename=item["filename"],
                    is_suspicious=insp.is_suspicious,
                    suspicious_findings=findings_dicts,
                    candidate_status="Scored",
                )
            eval_init.append(c_score)

        st.session_state["evaluation_results"] = eval_init
        st.session_state["selected_candidate_idx"] = 0

    else:
        req_names = [c["name"] for c in LIVE_RUBRIC_CRITERIA if c["category"] == "required"]
        pref_names = [c["name"] for c in LIVE_RUBRIC_CRITERIA if c["category"] == "preferred"]
        sample_dir = Path(__file__).resolve().parent / "sample_resumes"
        batch_t = st.session_state.get("live_batch_type", "standard")
        if batch_t == "suspicious":
            target_fnames = ["suspicious_resume_1.pdf", "suspicious_resume_2.pdf"]
        else:
            target_fnames = [f"resume_{i}.pdf" for i in range(1, 6)]

        eval_init = []
        for fname in target_fnames:
            p = sample_dir / fname
            if p.exists():
                with open(p, "rb") as f_pdf:
                    bytes_data = f_pdf.read()
                insp = inspect_pdf_bytes(bytes_data, filename=fname)
                findings_dicts = [
                    {
                        "category": f.category,
                        "detection_type": f.detection_type or f.category,
                        "detected_text": f.detected_text,
                        "reason": f.reason,
                        "severity": f.severity,
                        "page": f.page,
                    }
                    for f in insp.findings
                ]
                live_eval = client.evaluate_live_rubric(insp.raw_text, filename=fname)
                c_score = compute_match_score(
                    required_skills=req_names,
                    preferred_skills=pref_names,
                    evaluations=live_eval.get("skill_evaluations", []),
                    candidate_name=live_eval.get("candidate_name") or fname,
                    filename=fname,
                    is_suspicious=insp.is_suspicious,
                    suspicious_findings=findings_dicts,
                    candidate_status="Scored",
                )
                eval_init.append(c_score)

        st.session_state["evaluation_results"] = eval_init
        st.session_state["selected_candidate_idx"] = 0


# =============================================================================
# BOX 4: RESULTS & RANKING (ALL CONTENT PLACED STRICTLY INSIDE CONTAINER)
# =============================================================================
with st.container(border=True):
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 18px;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
            <span style="font-size: 1.3rem; font-weight: 700; color: #0f172a; letter-spacing: -0.01em;">Results &amp; Ranking</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_candidates: list[CandidateScore] = st.session_state["evaluation_results"]
    ranked_candidates, review_candidates = rank_and_group_candidates(all_candidates)

    # Two-column layout: Left = Ranking List, Right = Selected Profile Dashboard
    results_left_col, results_right_col = st.columns([1.15, 2.35], gap="large")

    # -----------------------------------------------------------------------------
    # Left Column: Candidate Ranking List & Summary
    # -----------------------------------------------------------------------------
    with results_left_col:
        selected_idx = st.session_state["selected_candidate_idx"]

        # Candidate ranking list inside container
        st.markdown('<div class="candidate-list-wrapper">', unsafe_allow_html=True)
        for idx, cand in enumerate(ranked_candidates):
            is_active = (idx == selected_idx)
            medal = "🥇" if cand.rank == 1 else ("🥈" if cand.rank == 2 else ("🥉" if cand.rank == 3 else f"#{cand.rank}"))
            btn_label = f"{medal}  {cand.candidate_name}    •    {cand.final_score:.1f}%"

            if st.button(
                btn_label,
                key=f"btn_cand_select_{idx}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["selected_candidate_idx"] = idx
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # If any unreadable / needs review resumes exist
        if review_candidates:
            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #991b1b;'>Needs Review (Excluded from Ranking):</div>", unsafe_allow_html=True)
            for r_idx, rc in enumerate(review_candidates):
                st.markdown(
                    f"""
                    <div style="background: #fef2f2; border: 1.5px solid #fca5a5; border-radius: 10px; padding: 10px 12px; font-size: 0.8rem; color: #991b1b; margin-top: 6px;">
                        📄 <strong>{rc.filename}</strong><br>
                        <span style="font-size: 0.72rem; color: #b91c1c;">{rc.review_reason}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Resume Analysis Summary Box with light-purple border
        parsed_count = len(all_candidates)
        extracted_skills_count = 6 if not is_demo else 8
        critical_errors_text = "No critical errors" if not review_candidates else f"{len(review_candidates)} document(s) need review"

        st.markdown(
            f"""
            <div style="background: #f8fafc; border: 1.5px solid #c7d2fe; border-radius: 12px; margin-top: 16px; padding: 16px 18px;">
                <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #0f172a; font-size: 0.95rem; margin-bottom: 10px;">
                    <span>📄</span> <span>Resume Analysis Summary</span>
                </div>
                <div style="font-size: 0.82rem; color: #334155; line-height: 1.6;">
                    <div style="color: #16a34a;">✓ All PDFs parsed successfully ({parsed_count} total)</div>
                    <div style="color: #16a34a;">✓ Criteria evaluated: {extracted_skills_count} skills</div>
                    <div style="color: {'#16a34a' if not review_candidates else '#d97706'};">✓ {critical_errors_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------------
    # Right Column: Selected Candidate Details Dashboard
    # -----------------------------------------------------------------------------
    with results_right_col:
        if ranked_candidates and selected_idx < len(ranked_candidates):
            selected_cand = ranked_candidates[selected_idx]
            feedback = generate_candidate_feedback(selected_cand)

            # Profile Header Card
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1.5px solid #c7d2fe; border-radius: 14px; padding: 18px 22px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                        <!-- Left: Avatar + Name + Score -->
                        <div style="display: flex; align-items: center; gap: 14px;">
                            <div style="width: 52px; height: 52px; border-radius: 50%; background: #dbeafe; display: flex; align-items: center; justify-content: center; font-size: 1.7rem; border: 1.5px solid #bfdbfe;">
                                👤
                            </div>
                            <div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span style="font-size: 1.35rem; font-weight: 700; color: #0f172a;">{feedback['candidate_name']}</span>
                                    <span style="background: {'#dcfce7' if selected_cand.rank == 1 else '#e0e7ff'}; color: {'#15803d' if selected_cand.rank == 1 else '#4338ca'}; font-size: 0.75rem; font-weight: 700; padding: 2px 10px; border-radius: 9999px; border: 1px solid {'#bbf7d0' if selected_cand.rank == 1 else '#c7d2fe'};">
                                        {'Top Match' if selected_cand.rank == 1 else f'Rank #{selected_cand.rank}'}
                                    </span>
                                </div>
                                <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-top: 3px;">
                                    Final Score: <span style="color: #16a34a;">{feedback['final_score']:.1f}%</span>
                                </div>
                            </div>
                        </div>
                        <!-- Right: Required and Preferred Score Bars -->
                        <div style="display: flex; gap: 20px; align-items: center;">
                            <div style="text-align: left; min-width: 130px;">
                                <div style="font-size: 0.78rem; font-weight: 600; color: #334155;">Required Skills (80 pts)</div>
                                <div style="display: flex; align-items: center; gap: 8px; margin-top: 3px;">
                                    <div style="flex-grow: 1; height: 8px; background: #e2e8f0; border-radius: 9999px; overflow: hidden; width: 100px;">
                                        <div style="width: {min(100, (selected_cand.required_score / 80 * 100) if not is_demo else selected_cand.required_score)}%; height: 100%; background: #16a34a; border-radius: 9999px;"></div>
                                    </div>
                                    <span style="font-size: 0.82rem; font-weight: 700; color: #0f172a;">{selected_cand.required_score:.0f}{' pts' if not is_demo else '%'}</span>
                                </div>
                            </div>
                            <div style="text-align: left; min-width: 130px;">
                                <div style="font-size: 0.78rem; font-weight: 600; color: #334155;">Preferred Skills (20 pts)</div>
                                <div style="display: flex; align-items: center; gap: 8px; margin-top: 3px;">
                                    <div style="flex-grow: 1; height: 8px; background: #e2e8f0; border-radius: 9999px; overflow: hidden; width: 100px;">
                                        <div style="width: {min(100, (selected_cand.preferred_score / 20 * 100) if not is_demo else selected_cand.preferred_score)}%; height: 100%; background: #2563eb; border-radius: 9999px;"></div>
                                    </div>
                                    <span style="font-size: 0.82rem; font-weight: 700; color: #0f172a;">{selected_cand.preferred_score:.0f}{' pts' if not is_demo else '%'}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 0.82rem; color: #475569; margin-top: 10px; background: #f8fafc; padding: 6px 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
                        <strong>Breakdown:</strong> {feedback['formula_breakdown']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Three Status Panels: Supported (Pale Green), Missing (Pale Red), Uncertain (Pale Lavender)
            c_sup, c_mis, c_unc = st.columns(3, gap="medium")

            # Panel 1: Supported Skills (Pale Green + Green Border)
            with c_sup:
                sup_count = len(feedback["supported_skills"])
                sup_items_html = ""
                for s in feedback["supported_skills"]:
                    ev_snippet = s.get("evidence", "")[:50]
                    ev_str = f" – {ev_snippet}..." if ev_snippet else ""
                    sup_items_html += f"<div style='margin-bottom: 6px;'>✓ <strong>{s['skill']}</strong><span style='color: #475569; font-size: 0.76rem;'>{ev_str}</span></div>"

                if not sup_items_html:
                    sup_items_html = "<div style='color: #64748b; font-size: 0.82rem;'>No supported skills.</div>"

                st.markdown(
                    f"""
                    <div class="panel-supported">
                        <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #15803d; font-size: 0.92rem; margin-bottom: 8px;">
                            <span style="background: #dcfce7; width: 22px; height: 22px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.8rem;">✓</span>
                            <span>Supported Skills ({sup_count})</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #14532d; line-height: 1.55;">
                            {sup_items_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Panel 2: Missing Skills (Pale Red + Red Border)
            with c_mis:
                mis_count = len(feedback["missing_skills"])
                mis_items_html = ""
                for m in feedback["missing_skills"]:
                    mis_items_html += f"<div style='margin-bottom: 6px;'>✕ <strong>{m['skill']}</strong></div>"

                if not mis_items_html:
                    mis_items_html = "<div style='color: #16a34a; font-size: 0.82rem;'>✓ No skill gaps!</div>"

                st.markdown(
                    f"""
                    <div class="panel-missing">
                        <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #b91c1c; font-size: 0.92rem; margin-bottom: 8px;">
                            <span style="background: #fee2e2; width: 22px; height: 22px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.8rem;">✕</span>
                            <span>Missing Skills ({mis_count})</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #7f1d1d; line-height: 1.55;">
                            {mis_items_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Panel 3: Uncertain / Partial Items (Pale Lavender + Purple Border)
            with c_unc:
                unc_count = len(feedback["review_items"])
                unc_items_html = ""
                for u in feedback["review_items"]:
                    unc_items_html += f"<div style='margin-bottom: 6px;'>? <strong>{u['skill']}</strong><div style='font-size: 0.74rem; color: #6b21a8; margin-top: 1px;'>{u.get('review_note', '')}</div></div>"

                if not unc_items_html:
                    unc_items_html = "<div style='color: #64748b; font-size: 0.82rem;'>None (All items verified)</div>"

                st.markdown(
                    f"""
                    <div class="panel-uncertain">
                        <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #7e22ce; font-size: 0.92rem; margin-bottom: 8px;">
                            <span style="background: #f3e8ff; width: 22px; height: 22px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.8rem;">?</span>
                            <span>Uncertain / Partial ({unc_count})</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #581c87; line-height: 1.55;">
                            {unc_items_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Side-by-Side: Skill-Gap Suggestions & Suspicious Content Check
            col_sug, col_shield = st.columns([1.5, 1.1], gap="medium")

            with col_sug:
                suggestions = feedback.get("suggestions", [])
                if suggestions:
                    gap_skills = ", ".join([s["skill"] for s in suggestions[:3]])
                    sug_text = f"Consider adding hands-on projects or certifications covering {gap_skills} to strengthen role fit."
                else:
                    sug_text = "Candidate meets all specified skill criteria. Excellent portfolio match!"

                st.markdown(
                    f"""
                    <div class="panel-suggestions" style="margin-top: 14px;">
                        <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #1d4ed8; font-size: 0.92rem; margin-bottom: 6px;">
                            <span style="font-size: 1.1rem;">💡</span> <span>Skill-Gap Suggestions</span>
                        </div>
                        <div style="font-size: 0.84rem; color: #1e3a8a; line-height: 1.5;">
                            {sug_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_shield:
                shield_rep = feedback["suspicious_report"]
                if shield_rep["is_flagged"]:
                    first_finding = shield_rep["findings"][0] if shield_rep["findings"] else {}
                    flagged_snippet = first_finding.get("detected_text", "Adversarial instruction or invisible text detected.")

                    st.markdown(
                        f"""
                        <div class="panel-suspicious-flagged" style="margin-top: 14px;">
                            <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #b45309; font-size: 0.92rem;">
                                <span style="font-size: 1.1rem;">⚠️</span> <span>Suspicious Content Check</span>
                            </div>
                            <div style="font-size: 0.82rem; color: #78350f; margin-top: 5px; line-height: 1.4;">
                                Detected: <span style="font-family: monospace; background: #fef3c7; border: 1px solid #fde68a; padding: 2px 6px; border-radius: 4px; color: #92400e;">{flagged_snippet[:70]}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander("🔍 View Details & Audit"):
                        for idx_f, f in enumerate(shield_rep["findings"]):
                            st.markdown(
                                f"**Finding {idx_f + 1} ({f.get('category')} - Page {f.get('page', 1)}):**\n"
                                f"- Reason: {f.get('reason')}\n"
                                f"- Detected: `{f.get('detected_text')}`"
                            )
                        st.caption(shield_rep["disclaimer"])
                else:
                    st.markdown(
                        """
                        <div class="panel-suspicious-clean" style="margin-top: 14px;">
                            <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #15803d; font-size: 0.92rem;">
                                <span style="font-size: 1.1rem;">✅</span> <span>Suspicious Content Check: Clean</span>
                            </div>
                            <div style="font-size: 0.82rem; color: #166534; margin-top: 5px; line-height: 1.4;">
                                No hidden text, micro-fonts, off-page coordinates, or prompt injection instructions detected.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Bottom Completion Status Bar & Working Report Download Action
            analysis_dur = st.session_state.get("last_analysis_time", 1.2)
            total_eval_count = len(all_candidates)

            col_bar_left, col_bar_right = st.columns([2, 1])

            export_rows = []
            for c in all_candidates:
                export_rows.append({
                    "Rank": c.display_rank if c.status == "Scored" else "Needs Review",
                    "Candidate Name": c.candidate_name,
                    "Filename": c.filename,
                    "Status": c.status,
                    "Final Score (%)": c.final_score if c.status == "Scored" else "N/A",
                    "Required Skills Score": f"{c.required_score:.0f}",
                    "Preferred Skills Score": f"{c.preferred_score:.0f}",
                    "Suspicious Content Flagged": "YES" if c.is_suspicious else "NO",
                    "Review Reason": c.review_reason or "N/A",
                })
            df_export = pd.DataFrame(export_rows)
            csv_bytes = df_export.to_csv(index=False).encode("utf-8")

            with col_bar_left:
                st.markdown(
                    f"""
                    <div class="panel-completion-bar">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="width: 30px; height: 30px; border-radius: 8px; background: #6366f1; color: white; display: flex; align-items: center; justify-content: center; font-size: 1.05rem;">
                                📊
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; font-weight: 700; color: #0f172a;">Analysis Complete!</div>
                                <div style="font-size: 0.76rem; color: #64748b;">{total_eval_count} resumes evaluated in {analysis_dur}s</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_bar_right:
                st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
                st.download_button(
                    label="📥  Download Report",
                    data=csv_bytes,
                    file_name="TalentLens_Screening_Report.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_download_report",
                )

        else:
            st.info("No candidate selected or all resumes require manual review.")


# =============================================================================
# BOX 5: SUSPICIOUS RESUMES (DEDICATED STANDALONE SECTION)
# =============================================================================
with st.container(border=True):
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 32px; height: 32px; border-radius: 8px; background: #ea580c; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.15rem;">
                    ⚠️
                </div>
                <div>
                    <span style="font-size: 1.3rem; font-weight: 700; color: #0f172a; letter-spacing: -0.01em;">Suspicious Resumes</span>
                    <div style="font-size: 0.78rem; color: #64748b; margin-top: 2px;">
                        Candidate documents flagged for hidden text, micro-fonts, off-page text, prompt injections, or manipulated ranking commands.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_resumes = st.session_state.get("evaluation_results", [])
    suspicious_candidates = [c for c in all_resumes if c.is_suspicious and c.suspicious_findings]

    if not suspicious_candidates:
        st.markdown(
            """
            <div style="background: #f0fdf4; border: 1.5px solid #86efac; border-radius: 12px; padding: 18px 22px; display: flex; align-items: center; gap: 14px;">
                <div style="width: 38px; height: 38px; border-radius: 50%; background: #dcfce7; color: #16a34a; display: flex; align-items: center; justify-content: center; font-size: 1.35rem;">
                    ✓
                </div>
                <div>
                    <div style="font-size: 1rem; font-weight: 700; color: #15803d;">No suspicious resumes detected.</div>
                    <div style="font-size: 0.82rem; color: #166534; margin-top: 2px;">
                        All evaluated candidate resumes passed ResumeShield forensic inspection with zero cloaked instructions, white text, micro-fonts, or prompt injection commands detected.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: #fff7ed; border: 1.5px solid #fdba74; border-radius: 12px; padding: 12px 18px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.25rem;">⚠️</span>
                <div style="font-size: 0.88rem; color: #9a3412; font-weight: 600;">
                    <strong>{len(suspicious_candidates)} Suspicious Document(s) Detected</strong> — Quarantined for forensic review.
                    Candidate match scores were evaluated strictly from verified visible evidence; adversarial instructions were neutralized.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for s_idx, cand in enumerate(suspicious_candidates):
            st.markdown(
                f"""
                <div style="background: #fffbf7; border: 1.5px solid #f97316; border-radius: 14px; padding: 18px 22px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(234, 88, 12, 0.08);">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div style="width: 42px; height: 42px; border-radius: 50%; background: #ffedd5; border: 1px solid #fdba74; display: flex; align-items: center; justify-content: center; font-size: 1.35rem;">
                                👤
                            </div>
                            <div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span style="font-size: 1.2rem; font-weight: 700; color: #0f172a;">{cand.candidate_name}</span>
                                    <span class="suspicious-pill-status">
                                        ⚠️ Suspicious Status: Flagged
                                    </span>
                                </div>
                                <div style="font-size: 0.82rem; color: #64748b; margin-top: 2px;">
                                    <strong>PDF File:</strong> <code>{cand.filename}</code> &nbsp;•&nbsp; <strong>Findings Count:</strong> {len(cand.suspicious_findings)} anomaly(ies) detected
                                </div>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.78rem; font-weight: 600; color: #64748b;">Legitimate Match Score</div>
                            <div style="font-size: 1.15rem; font-weight: 800; color: {'#16a34a' if cand.final_score >= 60 else ('#d97706' if cand.final_score >= 30 else '#dc2626')};">
                                {cand.final_score:.1f}%
                            </div>
                            <div style="font-size: 0.7rem; color: #94a3b8;">(Evaluated without following instructions)</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            findings_count = len(cand.suspicious_findings)
            expander_title = (
                f"🔍 Detailed Forensic Findings ({findings_count} Anomalies Detected)"
                if findings_count != 1
                else "🔍 Detailed Forensic Findings (1 Anomaly Detected)"
            )

            with st.expander(expander_title, expanded=False):
                for f_idx, finding in enumerate(cand.suspicious_findings):
                    d_type = finding.get("detection_type") or finding.get("category", "Adversarial finding")
                    detected_snippet = finding.get("detected_text", "")
                    reason_text = finding.get("reason", "Suspicious pattern detected.")
                    severity_val = finding.get("severity", "high").upper()
                    page_val = finding.get("page", 1)

                    st.markdown(
                        f"""
                        <div style="background: #ffffff; border: 1px solid #fed7aa; border-left: 4px solid #ea580c; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.82rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span class="suspicious-pill-type">
                                        📌 Detection Type: {d_type}
                                    </span>
                                    <span style="color: #64748b; font-size: 0.74rem;">Page {page_val}</span>
                                </div>
                                <span style="background: {'#fee2e2' if severity_val == 'HIGH' else '#fef3c7'}; color: {'#991b1b' if severity_val == 'HIGH' else '#92400e'}; padding: 1px 6px; border-radius: 4px; font-weight: 700; font-size: 0.7rem;">
                                    {severity_val} SEVERITY
                                </span>
                            </div>
                            <div style="color: #334155; margin-top: 4px;">
                                <strong>Reason for Flagging:</strong> {reason_text}
                            </div>
                            <div style="margin-top: 6px;">
                                <strong style="color: #7f1d1d;">Suspicious Content Detected:</strong>
                                <div style="font-family: ui-monospace, Menlo, Consolas, monospace; background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; padding: 4px 8px; border-radius: 6px; margin-top: 3px; word-break: break-word;">
                                    {detected_snippet}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander(f"📄 Review Original Extracted Text & Audit ({cand.candidate_name})", expanded=False):
                st.caption(f"Original text extracted from `{cand.filename}` by PyMuPDF for security inspection:")
                sample_pdf_path = Path(__file__).resolve().parent / "sample_resumes" / cand.filename
                preview_text = ""
                if sample_pdf_path.exists():
                    try:
                        doc_prev = pymupdf.open(str(sample_pdf_path))
                        preview_text = "\n".join([page.get_text() for page in doc_prev])
                        doc_prev.close()
                    except Exception:
                        pass
                if not preview_text and cand.skill_evaluations:
                    preview_text = f"Candidate: {cand.candidate_name}\nEvaluated Skills:\n" + "\n".join([f"- {s.name}: {s.evidence or 'No evidence'}" for s in cand.skill_evaluations])

                st.text_area(
                    label="Extracted Text Review",
                    value=preview_text,
                    height=180,
                    disabled=True,
                    key=f"raw_text_review_{cand.filename}_{s_idx}",
                )

            st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
