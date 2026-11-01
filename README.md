# TalentLens AI

> **Evaluate multiple resumes, rank candidates, and detect suspicious content.**  
> *Smarter Matching. Safer Screening.*

TalentLens AI is an automated resume-screening application that evaluates multiple candidate PDF resumes against the same job description using a strict, deterministic Python scoring formula and evidence-based verification. It includes a dedicated **Suspicious Content Check** that independently inspects PDF text styling, layout properties, and content for hidden text, micro-fonts, off-screen bounding boxes, and adversarial prompt injections—ensuring security findings remain separate from match scores and never bias automated ranking decisions.

---

## 🌟 Key Features

1. **Structured Job Criteria Parsing**:
   - Parses job descriptions into clearly defined **Required** and **Preferred** skills using Gemini (in Live Mode) or a local structured parser (in Demo Mode).
   - Validates that requirements are distinct and usable; prompts the user to revise vague or empty descriptions rather than hallucinating requirements.

2. **Suspicious Content Check (Forensic PDF Inspection)**:
   - Uses PyMuPDF (`fitz`) to inspect observable styling, font properties, and coordinates:
     - **Hidden White Text**: Detects text rendered in white/near-white RGB colors matching typical page backgrounds.
     - **Micro-Font Text**: Identifies keyword stuffing rendered at sizes `< 3.0pt`.
     - **Off-Page Coordinates**: Detects text placed outside visible page boundaries.
     - **Overlapping Text Spans**: Identifies text drawn directly on top of other text.
     - **Adversarial AI Instructions**: Scans extracted text for prompt injection commands (e.g. *"ignore previous instructions"*, *"award 100/100"*, *"override rubric"*).
   - **Critical Safety Guardrail:** Suspicious-content findings are reported in a separate audit section and **never** alter, penalize, or contaminate the candidate's match score.

3. **Deterministic Python Scoring (LLM Does NOT Score)**:
   - Scoring is computed strictly in Python code, never by the LLM:
     $$\text{Required Skill Score} = \left(\frac{\text{Supported Required Skills}}{\text{Total Required Skills}}\right) \times 100$$
     $$\text{Preferred Skill Score} = \left(\frac{\text{Supported Preferred Skills}}{\text{Total Preferred Skills}}\right) \times 100$$
     $$\text{Final Match Score} = 0.80 \times \text{Required Skill Score} + 0.20 \times \text{Preferred Skill Score}$$
   - When no preferred skills are specified:
     $$\text{Final Match Score} = \text{Required Skill Score}$$
   - Every supported skill requires direct verbatim evidence from the resume.
   - Ambiguous or uncertain evidence is flagged as "Needs review" and does not inflate the score.
   - Repeated keywords do not count more than once.

4. **Fair Ranking with Tied Score Handling**:
   - Ranks candidates by Final Match Score from highest to lowest.
   - Equal scores receive fair tied rankings (e.g., `#1 (Tie)`, `#1 (Tie)`, `#3`). No artificial tie-breakers are invented.

5. **Isolated "Needs Review" Queue**:
   - Resumes that are unreadable, empty, corrupted, or scanned images requiring OCR are separated into a "Needs Review" section, keeping the scored ranking reliable and trustworthy.

6. **Actionable Feedback & Skill-Gap Suggestions**:
   - Displays supported skills with resume evidence quotes, missing skills, uncertain items, and constructive, role-aligned skill improvement suggestions.

7. **Zero-API-Key Demo Mode**:
   - Fully functional out of the box with synthetic demo candidates and sample job requirements. Produces meaningfully varied, non-zero scores and full feature demonstration completely offline without requiring a Gemini API key.

---

## 🛠️ Tech Stack

- **Python 3.10+** (tested on Python 3.13)
- **Streamlit** (interactive web dashboard)
- **PyMuPDF (`pymupdf`)** (PDF extraction and forensic property inspection)
- **Google GenAI SDK (`google-genai`)** (Gemini 2.5 Flash for requirements and evidence structuring)
- **python-dotenv** (API key management)
- **pytest** (automated unit and integration testing)
- **pandas** (report generation and tabular export)

---

## 🚀 Installation & Quickstart

### 1. Clone or Open the Project
```powershell
cd "C:\Users\DIVYA DHARSHINI\.gemini\antigravity\scratch\talentlens-ai"
```

### 2. Set Up a Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🔑 Gemini API Key Configuration (Optional for Live Mode)

TalentLens AI works completely offline in **Demo Analysis** mode without an API key. To activate **Live Gemini Analysis**:

1. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Open `.env` and enter your Gemini API key from [Google AI Studio](https://aistudio.google.com/):
   ```env
   GEMINI_API_KEY=AIzaSyYourActualKeyHere
   ```
3. The `.env` file is automatically ignored by `.gitignore` to prevent committing secrets.

---

## ☁️ Streamlit Community Cloud Deployment

TalentLens AI is ready for instant 1-click deployment on [Streamlit Community Cloud](https://share.streamlit.io/):

1. **Push your repository to GitHub**:
   Ensure `.env` and `.streamlit/secrets.toml` are NOT committed (already configured in `.gitignore`).
2. **Deploy on Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
   - Select your repository, branch (`main`), and set the main file path to `app.py`.
3. **Configure Gemini API Key (Optional for Live Mode)**:
   - In your app dashboard on Streamlit Cloud, click **Settings** -> **Secrets**.
   - Paste your secret as shown in `.streamlit/secrets.toml.example`:
     ```toml
     GEMINI_API_KEY = "AIzaSy..."
     ```
   - Click **Save**. The app will automatically reload in Live Mode!
   - If no API key is provided, the app will continue to run seamlessly in **Demo Mode (Offline)** with synthetic sample resumes.

---

## 🏃 Running the Application

Launch the Streamlit web application:
```powershell
streamlit run app.py
```

Once running, navigate to the local URL (typically `http://localhost:8501`) in your web browser.

---

## 🧪 Running Automated Tests

Run the full pytest suite:
```powershell
pytest -v
```

All 30 unit and integration tests will execute, verifying:
- Deterministic scoring formula with and without preferred skills
- Missing and uncertain evidence handling
- Deduplication of skills
- Repeated keywords not increasing score
- Tied ranking preservation without artificial tie-breakers
- Suspicious content flags independence from match scores
- Collapsible Detailed Forensic Findings UI
- Suspicious Resumes filtering and anomaly reporting
- Extraction resilience when individual resumes fail
- Unreadable and zero-byte PDF handling
- Missing API key and malformed response handling
- Dual Gemini key loading from environment variables or st.secrets
- Demo mode varied scoring and correct candidate ranking
- Live mode calibrated targets (80/20 rubric)

Expected test summary:
```text
============================= 30 passed in 5.2s ==============================
```

---

## 📁 Project Architecture

```
talentlens-ai/
├── .env                      # Local secret API key (git-ignored)
├── .env.example              # Template environment file
├── .gitignore                # Protects secrets (.env, secrets.toml) and caches
├── requirements.txt          # Python dependencies (includes pandas)
├── README.md                 # Project documentation & deployment guide
├── app.py                    # Interactive Streamlit dashboard
├── .streamlit/
│   ├── config.toml           # Theme and cloud-ready server settings
│   └── secrets.toml.example  # Template for Streamlit Cloud Secrets
├── sample_resumes/           # Clean synthetic test PDF resumes
│   ├── resume_1.pdf ... resume_5.pdf
│   ├── suspicious_resume_1.pdf
│   └── suspicious_resume_2.pdf
├── src/
│   ├── __init__.py
│   ├── config.py             # Config thresholds, weights, dual API key loader
│   ├── pdf_inspector.py      # Suspicious Content Check & PyMuPDF extractor
│   ├── scorer.py             # Deterministic scoring formula & tied ranking
│   ├── gemini_client.py      # Untrusted-data safe LLM client & fallback
│   ├── feedback.py           # Candidate feedback & gap guidance generator
│   └── demo_data.py          # Synthetic job description & in-memory PDF generator
└── tests/
    ├── __init__.py
    ├── test_scorer.py            # Tests for scoring formula, ties, duplicates
    ├── test_resumeshield.py      # Tests for white text, micro-fonts, injection
    ├── test_suspicious_resumes.py# Tests for suspicious section & collapsible findings
    ├── test_live_scoring.py      # Tests for live rubric calibration
    └── test_workflow.py          # Resilience, secrets loading, and end-to-end tests
```

---

## ⚖️ Limitations & Ethical Standards

- **Scoring Purpose**: The match score measures alignment with explicit job requirements; it does not measure overall human intelligence, character, or predict on-the-job success.
- **Imperfect Detection**: The Suspicious Content Check uses observable formatting heuristics and pattern matching. It is not foolproof; novel cloaking methods may evade detection, and non-standard resume designs might trigger false indicators.
- **Fair Treatment**: Security findings are advisory audits for human recruiters. Candidates are **never** automatically disqualified or penalized in their score based on detected flags.
- **Human-in-the-Loop**: All hiring, shortlisting, and interview decisions must be made by qualified human professionals.
