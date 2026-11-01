"""Automated tests for the separate Suspicious Resumes section."""

import pytest
import pymupdf
from pathlib import Path

from src.pdf_inspector import inspect_pdf_bytes, InspectionResult
from src.scorer import (
    evaluate_data_analyst_rubric_evidence,
    compute_match_score,
    rank_and_group_candidates,
    CandidateScore,
)
from src.gemini_client import LIVE_RUBRIC_CRITERIA, GeminiClient
from src.demo_data import get_demo_resumes


def test_suspicious_resume_1_detection():
    """Verify: suspicious_resume_1.pdf is flagged, has expected detection types, and score is 90% (not 100%)."""
    pdf_path = Path(__file__).resolve().parent.parent / "sample_resumes" / "suspicious_resume_1.pdf"
    assert pdf_path.exists(), "suspicious_resume_1.pdf must exist in sample_resumes"

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    insp: InspectionResult = inspect_pdf_bytes(pdf_bytes, filename="suspicious_resume_1.pdf")
    assert insp.status == "OK"
    assert insp.is_suspicious is True
    assert len(insp.findings) > 0

    # Check candidate name and score
    cand_name, evaluations = evaluate_data_analyst_rubric_evidence(insp.raw_text, "suspicious_resume_1.pdf")
    assert cand_name == "Kavya Menon-Ashworth"

    req_names = [c["name"] for c in LIVE_RUBRIC_CRITERIA if c["category"] == "required"]
    pref_names = [c["name"] for c in LIVE_RUBRIC_CRITERIA if c["category"] == "preferred"]

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

    score = compute_match_score(
        required_skills=req_names,
        preferred_skills=pref_names,
        evaluations=evaluations,
        candidate_name=cand_name,
        filename="suspicious_resume_1.pdf",
        is_suspicious=insp.is_suspicious,
        suspicious_findings=findings_dicts,
        candidate_status="Scored",
    )

    # Legitimate score must be 90% (not the requested 100%)
    assert score.final_score == 90.0
    assert score.is_suspicious is True
    assert len(score.suspicious_findings) >= 3

    # Check detection types present
    types_found = {f["detection_type"] for f in score.suspicious_findings}
    assert any("manipulate candidate ranking" in t.lower() or "score manipulation" in t.lower() for t in types_found)
    assert any("hidden white text" in t.lower() for t in types_found)
    assert any("prompt injection" in t.lower() for t in types_found)


def test_suspicious_resume_2_detection():
    """Verify: suspicious_resume_2.pdf is flagged, has multiple adversarial detection types, and score is 0% (not 100%)."""
    pdf_path = Path(__file__).resolve().parent.parent / "sample_resumes" / "suspicious_resume_2.pdf"
    assert pdf_path.exists(), "suspicious_resume_2.pdf must exist in sample_resumes"

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    insp: InspectionResult = inspect_pdf_bytes(pdf_bytes, filename="suspicious_resume_2.pdf")
    assert insp.status == "OK"
    assert insp.is_suspicious is True
    assert len(insp.findings) >= 5

    cand_name, evaluations = evaluate_data_analyst_rubric_evidence(insp.raw_text, "suspicious_resume_2.pdf")
    assert cand_name == "Sebastian Holloway"

    req_names = [c["name"] for c in LIVE_RUBRIC_CRITERIA if c["category"] == "required"]
    pref_names = [c["name"] for c in LIVE_RUBRIC_CRITERIA if c["category"] == "preferred"]

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

    score = compute_match_score(
        required_skills=req_names,
        preferred_skills=pref_names,
        evaluations=evaluations,
        candidate_name=cand_name,
        filename="suspicious_resume_2.pdf",
        is_suspicious=insp.is_suspicious,
        suspicious_findings=findings_dicts,
        candidate_status="Scored",
    )

    # Legitimate score must be 0% (claims of being "Expert in Python/SQL" are unverified claims)
    assert score.final_score == 0.0
    assert score.is_suspicious is True

    # Check detection types present
    types_found = {f["detection_type"] for f in score.suspicious_findings}
    assert any("hidden white text" in t.lower() for t in types_found)
    assert any("small-font" in t.lower() or "micro" in t.lower() for t in types_found)
    assert any("prompt injection" in t.lower() for t in types_found)
    assert any("manipulate candidate ranking" in t.lower() for t in types_found)
    assert any("without supporting evidence" in t.lower() for t in types_found)


def test_normal_resume_not_in_suspicious_list():
    """Verify: Legitimate resumes (resume_1.pdf through resume_5.pdf) have is_suspicious=False and 0 findings."""
    resumes_dir = Path(__file__).resolve().parent.parent / "sample_resumes"

    for i in range(1, 6):
        pdf_path = resumes_dir / f"resume_{i}.pdf"
        assert pdf_path.exists()
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        insp = inspect_pdf_bytes(pdf_bytes, filename=f"resume_{i}.pdf")
        assert insp.status == "OK"
        assert insp.is_suspicious is False, f"Normal resume resume_{i}.pdf must not be flagged"
        assert len(insp.findings) == 0


def test_suspicious_resumes_filtering_logic():
    """Verify: Only suspicious candidates appear in the filtered suspicious list."""
    resumes_dir = Path(__file__).resolve().parent.parent / "sample_resumes"
    req_names = [c["name"] for c in LIVE_RUBRIC_CRITERIA if c["category"] == "required"]
    pref_names = [c["name"] for c in LIVE_RUBRIC_CRITERIA if c["category"] == "preferred"]

    all_candidates: list[CandidateScore] = []

    for name in ["resume_1.pdf", "suspicious_resume_1.pdf", "resume_4.pdf", "suspicious_resume_2.pdf"]:
        pdf_path = resumes_dir / name
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        insp = inspect_pdf_bytes(pdf_bytes, filename=name)
        cand_name, evaluations = evaluate_data_analyst_rubric_evidence(insp.raw_text, name)
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
        score = compute_match_score(
            required_skills=req_names,
            preferred_skills=pref_names,
            evaluations=evaluations,
            candidate_name=cand_name,
            filename=name,
            is_suspicious=insp.is_suspicious,
            suspicious_findings=findings_dicts,
            candidate_status="Scored",
        )
        all_candidates.append(score)

    # Filter for suspicious section
    suspicious_list = [c for c in all_candidates if c.is_suspicious and c.suspicious_findings]

    # Exactly the two suspicious resumes must be in the suspicious list
    assert len(suspicious_list) == 2
    suspicious_filenames = [c.filename for c in suspicious_list]
    assert "suspicious_resume_1.pdf" in suspicious_filenames
    assert "suspicious_resume_2.pdf" in suspicious_filenames

    # Normal resumes must NOT be in the suspicious list
    assert "resume_1.pdf" not in suspicious_filenames
    assert "resume_4.pdf" not in suspicious_filenames


def test_demo_mode_suspicious_filtering():
    """Verify: In Demo Mode, only Morgan Vance appears in suspicious resumes."""
    demo_items = get_demo_resumes()
    all_demo_scores = []
    client = GeminiClient(api_key=None)

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
        if insp.status == "OK":
            evidence = client.extract_resume_evidence(insp.raw_text, ["Python"], [], filename=item["filename"])
            score = compute_match_score(
                required_skills=["Python"],
                preferred_skills=[],
                evaluations=evidence.get("skill_evaluations", []),
                candidate_name=item["candidate_name"],
                filename=item["filename"],
                is_suspicious=insp.is_suspicious,
                suspicious_findings=findings_dicts,
                candidate_status="Scored",
            )
        else:
            score = compute_match_score(
                required_skills=["Python"],
                preferred_skills=[],
                evaluations=[],
                candidate_name=item["candidate_name"],
                filename=item["filename"],
                is_suspicious=insp.is_suspicious,
                suspicious_findings=findings_dicts,
                candidate_status="Needs review",
                review_reason=insp.review_reason,
            )
        all_demo_scores.append(score)

    suspicious_list = [c for c in all_demo_scores if c.is_suspicious and c.suspicious_findings]
    assert len(suspicious_list) == 1
    assert suspicious_list[0].candidate_name == "Morgan Vance"
    assert "Morgan_Vance" in suspicious_list[0].filename


def test_collapsible_detailed_forensic_findings():
    """Verify: Collapsible Detailed Forensic Findings heading and anomaly counters for both suspicious resumes."""
    resumes_dir = Path(__file__).resolve().parent.parent / "sample_resumes"

    # 1. suspicious_resume_1.pdf -> 12 anomalies
    with open(resumes_dir / "suspicious_resume_1.pdf", "rb") as f:
        insp1 = inspect_pdf_bytes(f.read(), filename="suspicious_resume_1.pdf")
    count_1 = len(insp1.findings)
    assert count_1 == 12
    title_1 = (
        f"🔍 Detailed Forensic Findings ({count_1} Anomalies Detected)"
        if count_1 != 1
        else "🔍 Detailed Forensic Findings (1 Anomaly Detected)"
    )
    assert title_1 == "🔍 Detailed Forensic Findings (12 Anomalies Detected)"

    # 2. suspicious_resume_2.pdf -> 53 anomalies
    with open(resumes_dir / "suspicious_resume_2.pdf", "rb") as f:
        insp2 = inspect_pdf_bytes(f.read(), filename="suspicious_resume_2.pdf")
    count_2 = len(insp2.findings)
    assert count_2 == 53
    title_2 = (
        f"🔍 Detailed Forensic Findings ({count_2} Anomalies Detected)"
        if count_2 != 1
        else "🔍 Detailed Forensic Findings (1 Anomaly Detected)"
    )
    assert title_2 == "🔍 Detailed Forensic Findings (53 Anomalies Detected)"
