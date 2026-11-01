"""Automated verification test for Part 2 & Part 3: Live Mode scoring rubric calibration."""

import pytest
import pymupdf
from pathlib import Path

from src.scorer import (
    evaluate_data_analyst_rubric_evidence,
    compute_match_score,
    rank_and_group_candidates,
)
from src.gemini_client import LIVE_RUBRIC_CRITERIA, JUNIOR_DATA_ANALYST_JOB_DESCRIPTION


def test_live_mode_calibration_targets():
    """Verify that all 5 sample resumes match the reference calibration targets:
    - Priya Nandakumar: 100%
    - Tobias Lindqvist: 85%
    - Elena Vasquez-Ruiz: 53%
    - Marcus Oyelaran: 45%
    - Jamal Whitfield: 13%
    """
    resumes_dir = Path(__file__).resolve().parent.parent / "sample_resumes"
    assert resumes_dir.exists(), "sample_resumes directory must exist"

    expected_targets = {
        "Priya Nandakumar": 100.0,
        "Tobias Lindqvist": 85.0,
        "Elena Vasquez-Ruiz": 53.0,
        "Marcus Oyelaran": 45.0,
        "Jamal Whitfield": 13.0,
    }

    req_names = [c["name"] for c in LIVE_RUBRIC_CRITERIA if c["category"] == "required"]
    pref_names = [c["name"] for c in LIVE_RUBRIC_CRITERIA if c["category"] == "preferred"]

    candidate_scores = []

    for i in range(1, 6):
        pdf_path = resumes_dir / f"resume_{i}.pdf"
        assert pdf_path.exists(), f"{pdf_path} must exist"

        doc = pymupdf.open(str(pdf_path))
        text = "\n".join([page.get_text() for page in doc])
        doc.close()

        cand_name, evaluations = evaluate_data_analyst_rubric_evidence(text, f"resume_{i}.pdf")
        score = compute_match_score(
            required_skills=req_names,
            preferred_skills=pref_names,
            evaluations=evaluations,
            candidate_name=cand_name,
            filename=f"resume_{i}.pdf",
        )
        candidate_scores.append(score)

    ranked, needs_review = rank_and_group_candidates(candidate_scores)

    assert len(ranked) == 5
    assert len(needs_review) == 0

    scores_by_name = {c.candidate_name: c.final_score for c in ranked}

    for name, expected_score in expected_targets.items():
        assert name in scores_by_name, f"Candidate {name} must be evaluated"
        assert scores_by_name[name] == expected_score, (
            f"Candidate {name} received {scores_by_name[name]}%, expected {expected_score}%"
        )

    # Verify ranking order: Priya > Tobias > Elena > Marcus > Jamal
    assert ranked[0].candidate_name == "Priya Nandakumar"
    assert ranked[0].rank == 1
    assert ranked[1].candidate_name == "Tobias Lindqvist"
    assert ranked[1].rank == 2
    assert ranked[2].candidate_name == "Elena Vasquez-Ruiz"
    assert ranked[2].rank == 3
    assert ranked[3].candidate_name == "Marcus Oyelaran"
    assert ranked[3].rank == 4
    assert ranked[4].candidate_name == "Jamal Whitfield"
    assert ranked[4].rank == 5
