"""Automated end-to-end workflow tests for TalentLens AI."""

import pytest
from src.pdf_inspector import inspect_pdf_bytes
from src.gemini_client import GeminiClient
from src.scorer import compute_match_score, rank_and_group_candidates
from src.feedback import generate_candidate_feedback
from src.demo_data import get_demo_resumes, SAMPLE_JOB_DESCRIPTION


def test_batch_processing_resilience_one_failing_resume():
    """Verify: If one resume is corrupted/unreadable, the system processes all other resumes without crashing."""
    valid_pdf_1 = get_demo_resumes()[0]["pdf_bytes"]  # Alex Rivera
    corrupted_pdf = b"GARBAGE_CORRUPTED_PDF_BYTES"
    valid_pdf_2 = get_demo_resumes()[1]["pdf_bytes"]  # Jordan Taylor

    batch = [
        ("alex.pdf", valid_pdf_1),
        ("broken.pdf", corrupted_pdf),
        ("jordan.pdf", valid_pdf_2),
    ]

    client = GeminiClient(api_key=None)  # Offline fallback
    parsed_jd = client.parse_job_description(SAMPLE_JOB_DESCRIPTION)
    req = parsed_jd["required_skills"]
    pref = parsed_jd["preferred_skills"]

    candidate_scores = []

    for fname, pdf_bytes in batch:
        # Step 1: Inspect PDF
        insp = inspect_pdf_bytes(pdf_bytes, filename=fname)

        if insp.status != "OK":
            score = compute_match_score(
                required_skills=req,
                preferred_skills=pref,
                evaluations=[],
                candidate_name=fname,
                filename=fname,
                is_suspicious=insp.is_suspicious,
                suspicious_findings=[f.to_dict() if hasattr(f, "to_dict") else vars(f) for f in insp.findings],
                candidate_status="Needs review",
                review_reason=insp.review_reason,
            )
        else:
            evidence = client.extract_resume_evidence(insp.raw_text, req, pref, filename=fname)
            score = compute_match_score(
                required_skills=req,
                preferred_skills=pref,
                evaluations=evidence["skill_evaluations"],
                candidate_name=evidence["candidate_name"],
                filename=fname,
                is_suspicious=insp.is_suspicious,
                suspicious_findings=[f.to_dict() if hasattr(f, "to_dict") else vars(f) for f in insp.findings],
            )
        candidate_scores.append(score)

    ranked, needs_review = rank_and_group_candidates(candidate_scores)

    # 2 valid resumes should be ranked, 1 broken resume should be in needs_review
    assert len(ranked) == 2
    assert len(needs_review) == 1
    assert needs_review[0].filename == "broken.pdf"
    assert "Malformed or corrupted" in needs_review[0].review_reason or "error" in needs_review[0].review_reason.lower()


def test_missing_api_key_graceful_handling():
    """Verify: When GEMINI_API_KEY is missing or invalid, client does not crash and provides clear state."""
    client = GeminiClient(api_key="")
    assert client.is_available() is False

    # Parsing job description falls back gracefully
    result = client.parse_job_description(SAMPLE_JOB_DESCRIPTION)
    assert result["is_valid"] is True
    assert len(result["required_skills"]) > 0

    # Extracting evidence falls back gracefully
    evidence = client.extract_resume_evidence(
        "Experience with Python and SQL in backend services.",
        ["Python", "SQL", "Docker"],
        ["Streamlit"],
    )
    assert "candidate_name" in evidence
    assert len(evidence["skill_evaluations"]) == 4


def test_empty_or_vague_job_description_validation():
    """Verify: Empty or vague job descriptions return is_valid=False with clarification message."""
    client = GeminiClient(api_key=None)

    # Empty string
    empty_result = client.parse_job_description("")
    assert empty_result["is_valid"] is False
    assert len(empty_result["clarification_message"]) > 0

    # Meaningless text
    vague_result = client.parse_job_description("We want someone good.")
    assert vague_result["is_valid"] is False
    assert "revise" in vague_result["clarification_message"].lower() or "empty" in vague_result["clarification_message"].lower() or "requirements" in vague_result["clarification_message"].lower()


def test_complete_demo_workflow_end_to_end():
    """Verify: The entire Demo Mode workflow runs cleanly and generates feedback for all synthetic resumes."""
    demo_resumes = get_demo_resumes()
    assert len(demo_resumes) == 5

    client = GeminiClient(api_key=None)
    parsed_jd = client.parse_job_description(SAMPLE_JOB_DESCRIPTION)
    req = parsed_jd["required_skills"]
    pref = parsed_jd["preferred_skills"]

    candidate_scores = []
    for item in demo_resumes:
        insp = inspect_pdf_bytes(item["pdf_bytes"], filename=item["filename"])
        if insp.status != "OK":
            score = compute_match_score(
                required_skills=req,
                preferred_skills=pref,
                evaluations=[],
                candidate_name=item["candidate_name"],
                filename=item["filename"],
                is_suspicious=insp.is_suspicious,
                suspicious_findings=[vars(f) for f in insp.findings],
                candidate_status="Needs review",
                review_reason=insp.review_reason,
            )
        else:
            evidence = client.extract_resume_evidence(
                insp.raw_text, req, pref, filename=item["filename"]
            )
            score = compute_match_score(
                required_skills=req,
                preferred_skills=pref,
                evaluations=evidence["skill_evaluations"],
                candidate_name=item["candidate_name"],
                filename=item["filename"],
                is_suspicious=insp.is_suspicious,
                suspicious_findings=[vars(f) for f in insp.findings],
            )
        candidate_scores.append(score)

    ranked, needs_review = rank_and_group_candidates(candidate_scores)

    # 4 scored candidates + 1 needs_review candidate
    assert len(ranked) == 4
    assert len(needs_review) == 1

    # Verify feedback generation works on every candidate
    for cand in ranked + needs_review:
        fb = generate_candidate_feedback(cand)
        assert fb["candidate_name"] is not None
        assert "suggestions" in fb
        assert "suspicious_report" in fb


def test_demo_mode_meaningful_varied_scores_and_ranking():
    """Verify: Demo Mode produces meaningfully varied scores, supported/missing/uncertain skills, and flagged injections."""
    demo_resumes = get_demo_resumes()
    client = GeminiClient(api_key=None)
    parsed_jd = client.parse_job_description(SAMPLE_JOB_DESCRIPTION)
    req = parsed_jd["required_skills"]
    pref = parsed_jd["preferred_skills"]

    candidate_scores = []
    for item in demo_resumes:
        insp = inspect_pdf_bytes(item["pdf_bytes"], filename=item["filename"])
        if insp.status != "OK":
            score = compute_match_score(
                required_skills=req,
                preferred_skills=pref,
                evaluations=[],
                candidate_name=item["candidate_name"],
                filename=item["filename"],
                is_suspicious=insp.is_suspicious,
                suspicious_findings=[vars(f) for f in insp.findings],
                candidate_status="Needs review",
                review_reason=insp.review_reason,
            )
        else:
            evidence = client.extract_resume_evidence(
                insp.raw_text, req, pref, filename=item["filename"]
            )
            score = compute_match_score(
                required_skills=req,
                preferred_skills=pref,
                evaluations=evidence["skill_evaluations"],
                candidate_name=item["candidate_name"],
                filename=item["filename"],
                is_suspicious=insp.is_suspicious,
                suspicious_findings=[vars(f) for f in insp.findings],
            )
        candidate_scores.append(score)

    ranked, needs_review = rank_and_group_candidates(candidate_scores)

    # 1. Scores must be sorted descending and all distinct / non-zero
    scores = [c.final_score for c in ranked]
    assert len(scores) == 4
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == 4  # All 4 scores are distinct
    assert all(s > 0 for s in scores)  # None are zero

    # 2. Top candidate has highest score
    assert ranked[0].candidate_name == "Alex Rivera"
    assert ranked[0].final_score > 90.0

    # 3. Uncertain items exist in demo data
    all_evals = []
    for c in ranked:
        all_evals.extend(c.skill_evaluations)
    uncertain_evals = [e for e in all_evals if e.status == "needs_review"]
    supported_evals = [e for e in all_evals if e.status == "supported"]
    missing_evals = [e for e in all_evals if e.status == "missing"]
    assert len(uncertain_evals) >= 1, "Demo mode must include examples of uncertain skills"
    assert len(supported_evals) >= 1, "Demo mode must include examples of supported skills"
    assert len(missing_evals) >= 1, "Demo mode must include examples of missing skills"

    # 4. Adversarial candidate is flagged, but score is based on genuine evidence (32.0%)
    morgan = next(c for c in ranked if "Morgan" in c.candidate_name)
    assert morgan.is_suspicious is True
    assert len(morgan.suspicious_findings) > 0
    assert morgan.final_score == 32.0  # NOT 100% as requested by prompt injection!

    # 5. Scanned document is quarantined in needs_review
    assert len(needs_review) == 1
    assert "Scan" in needs_review[0].candidate_name or "scanned" in needs_review[0].filename.lower()


def test_invalid_gemini_json_response_handling():
    """Verify: GeminiClient gracefully handles malformed/non-JSON responses with heuristic fallback."""
    from unittest.mock import MagicMock

    client = GeminiClient(api_key="mock_key_for_test")
    # Mock models.generate_content returning invalid JSON
    mock_resp = MagicMock()
    mock_resp.text = "NOT_VALID_JSON: {broken syntax"
    client._client = MagicMock()
    client._client.models.generate_content.return_value = mock_resp

    # Should fall back gracefully to heuristic job parser
    parsed = client.parse_job_description(SAMPLE_JOB_DESCRIPTION)
    assert parsed["is_valid"] is True
    assert "warning" in parsed or len(parsed["required_skills"]) > 0

    # Should fall back gracefully to heuristic evidence extractor
    extracted = client.extract_resume_evidence(
        "Candidate with 5 years of Python experience.",
        ["Python"],
        [],
    )
    assert extracted["candidate_name"] is not None
    assert len(extracted["skill_evaluations"]) == 1
    assert extracted["skill_evaluations"][0].status == "supported"


def test_gemini_api_key_loading_secrets_and_env():
    """Verify: API key loading supports both .env and Streamlit secrets with fallback."""
    import os
    import streamlit as st
    from src.config import get_gemini_api_key, is_api_key_configured

    orig_env = os.environ.get("GEMINI_API_KEY")
    try:
        # 1. Fallback to st.secrets when environment variable is unset
        os.environ.pop("GEMINI_API_KEY", None)
        st.secrets = {"GEMINI_API_KEY": "cloud_secret_test_key_12345"}
        assert get_gemini_api_key() == "cloud_secret_test_key_12345"
        assert is_api_key_configured() is True

        # 2. Environment variable takes precedence when explicitly set
        os.environ["GEMINI_API_KEY"] = "local_env_override_key_9999"
        assert get_gemini_api_key() == "local_env_override_key_9999"
        assert is_api_key_configured() is True
    finally:
        if orig_env is not None:
            os.environ["GEMINI_API_KEY"] = orig_env
        else:
            os.environ.pop("GEMINI_API_KEY", None)

