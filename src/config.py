"""Configuration settings and environment variable management for TalentLens AI."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically look for .env in current directory and project root
load_dotenv()
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

REQUIRED_SKILL_WEIGHT: float = 0.80
PREFERRED_SKILL_WEIGHT: float = 0.20

# ResumeShield Forensics Thresholds
MIN_EXTRACTABLE_TEXT_CHARS: int = 30
MICRO_FONT_SIZE_THRESHOLD: float = 3.0
WHITE_COLOR_THRESHOLD: int = 245  # (R, G, B each >= 245 considered white/near-white)

# Default Gemini model
DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"


def get_gemini_api_key() -> str:
    """Retrieve the Gemini API key safely from environment variables or Streamlit secrets.
    
    Compatible with:
    - Local development via .env (python-dotenv) or system environment variable
    - Streamlit Community Cloud deployment via st.secrets["GEMINI_API_KEY"]
    - Local secrets file via .streamlit/secrets.toml
    
    Returns:
        str: Clean API key string, or empty string if not found.
    """
    # 1. First check environment variables (local .env or system env)
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key and env_key != "your_key_here":
        return env_key

    # 2. Next check Streamlit secrets (Streamlit Community Cloud or .streamlit/secrets.toml)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                secret_key = str(st.secrets["GEMINI_API_KEY"]).strip()
                if secret_key and secret_key != "your_key_here":
                    return secret_key
    except Exception:
        # Gracefully handle cases where st.secrets is unavailable (headless CLI, tests, etc.)
        pass

    return env_key



def is_api_key_configured() -> bool:
    """Check whether a non-placeholder Gemini API key is configured."""
    key = get_gemini_api_key()
    return bool(key) and key != "your_key_here" and len(key) > 8
