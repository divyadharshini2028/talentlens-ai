"""Automated unit tests for the deterministic scoring formula and ranking engine."""

import pytest
from src.scorer import (
    compute_match_score,
    rank_and_group_candidates,
    SkillEvaluation,
    CandidateScore,
)


def test_score_formula_with_preferred_skills():
    """Verify: Final Match Score = 0.80 * Required Skill Score + 0.20 * Preferred Skill Score."""
    required_skills = ["Python", "SQL", "Docker", "Pytest", "FastAPI"]  # 5 required
    preferred_skills = ["Streamlit", "AWS"]  # 2 preferred

    # Candidate meets 4/5 required (80.0%) and 1/2 preferred (50.0%)
    # Expected: 0.80 * 80.0 + 0.20 * 50.0 = 64.0 + 10.0 = 74.0
    evaluations = [
        SkillEvaluation(name="Python", category="required", status="supported", evidence="5 yrs Python"),
        SkillEvaluation(name="SQL", category="required", status="supported", evidence="PostgreSQL experience"),
        SkillEvaluation(name="Docker", category="required", status="supported", evidence="Containerized apps"),
        SkillEvaluation(name="Pytest", category="required", status="supported", evidence="Automated testing with pytest"),
        SkillEvaluation(name="FastAPI", category="required", status="missing", evidence=None),
        SkillEvaluation(name="Streamlit", category="preferred", status="supported", evidence="Built Streamlit UI"),
        SkillEvaluation(name="AWS", category="preferred", status="missing", evidence=None),
    ]

    result = compute_match_score(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        evaluations=evaluations,
        candidate_name="Alice Smith",
        filename="alice.pdf",
    )

    assert result.status == "Scored"
    assert result.required_total == 5
    assert result.required_supported == 4
    assert result.required_score == 80.0
    assert result.preferred_total == 2
    assert result.preferred_supported == 1
    assert result.preferred_score == 50.0
    assert result.final_score == 74.0
    assert "0.80 ×" in result.formula_breakdown


def test_score_formula_with_no_preferred_skills():
    """Verify: If there are no preferred skills, Final Match Score = Required Skill Score."""
    required_skills = ["Python", "SQL", "Git", "Linux"]  # 4 required
    preferred_skills = []  # 0 preferred

    # Candidate meets 3/4 required = 75.0%
    evaluations = [
        SkillEvaluation(name="Python", category="required", status="supported", evidence="Core Python"),
        SkillEvaluation(name="SQL", category="required", status="supported", evidence="Complex queries"),
        SkillEvaluation(name="Git", category="required", status="supported", evidence="Git workflow"),
        SkillEvaluation(name="Linux", category="required", status="missing", evidence=None),
    ]

    result = compute_match_score(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        evaluations=evaluations,
        candidate_name="Bob Jones",
        filename="bob.pdf",
    )

    assert result.status == "Scored"
    assert result.has_preferred_skills is False
    assert result.required_total == 4
    assert result.required_supported == 3
    assert result.required_score == 75.0
    assert result.final_score == 75.0
    assert "no preferred skills" in result.formula_breakdown


def test_missing_and_uncertain_evidence_handling():
    """Verify: Skills with missing or uncertain ('needs_review') evidence do NOT count as supported."""
    required_skills = ["Python", "Machine Learning", "Kubernetes"]
    preferred_skills = []

    # 1 supported, 1 needs_review (uncertain), 1 missing
    evaluations = [
        SkillEvaluation(
            name="Python",
            category="required",
            status="supported",
            evidence="Proficient in Python.",
        ),
        SkillEvaluation(
            name="Machine Learning",
            category="required",
            status="needs_review",
            evidence="Read some ML books.",
            review_note="Theoretical interest only; insufficient hands-on evidence.",
        ),
        SkillEvaluation(
            name="Kubernetes",
            category="required",
            status="missing",
            evidence=None,
        ),
    ]

    result = compute_match_score(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        evaluations=evaluations,
        candidate_name="Charlie Brown",
        filename="charlie.pdf",
    )

    assert result.required_total == 3
    assert result.required_supported == 1  # Only Python counts
    assert result.final_score == round((1 / 3) * 100, 2)  # 33.33%


def test_duplicate_skills_do_not_count_more_than_once():
    """Verify: Skills mentioned multiple times or duplicated in requirements/evaluations count only once."""
    # Duplicate entries in job requirements
    required_skills = ["Python", "python", " PYTHON ", "SQL"]
    preferred_skills = ["Docker", "docker"]

    # Candidate repeated Python multiple times in evaluation
    evaluations = [
        SkillEvaluation(name="Python", category="required", status="supported", evidence="Python 1"),
        SkillEvaluation(name="python", category="required", status="supported", evidence="Python 2"),
        SkillEvaluation(name="SQL", category="required", status="supported", evidence="SQL query"),
        SkillEvaluation(name="Docker", category="preferred", status="supported", evidence="Docker container"),
        SkillEvaluation(name="docker", category="preferred", status="supported", evidence="Docker image"),
    ]

    result = compute_match_score(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        evaluations=evaluations,
        candidate_name="Dave Miller",
        filename="dave.pdf",
    )

    # Deduplicated totals: 2 required (Python, SQL), 1 preferred (Docker)
    assert result.required_total == 2
    assert result.required_supported == 2
    assert result.preferred_total == 1
    assert result.preferred_supported == 1
    assert result.final_score == 100.0


def test_repeated_keywords_do_not_inflate_score():
    """Verify: Repeated keywords or mentions of the same skill do not increase candidate match score."""
    required_skills = ["Python", "SQL", "Docker"]
    preferred_skills = []

    # Candidate with single Python evidence vs candidate with Python repeated 10 times
    single_eval = [
        SkillEvaluation(name="Python", category="required", status="supported", evidence="Built apps with Python"),
        SkillEvaluation(name="SQL", category="required", status="missing", evidence=None),
        SkillEvaluation(name="Docker", category="required", status="missing", evidence=None),
    ]

    repeated_eval = [
        SkillEvaluation(name="Python", category="required", status="supported", evidence="Built apps with Python"),
        SkillEvaluation(name="Python", category="required", status="supported", evidence="More Python code"),
        SkillEvaluation(name="Python", category="required", status="supported", evidence="Python Python Python"),
        SkillEvaluation(name="SQL", category="required", status="missing", evidence=None),
        SkillEvaluation(name="Docker", category="required", status="missing", evidence=None),
    ]

    score_single = compute_match_score(required_skills, preferred_skills, single_eval, "Single", "single.pdf")
    score_repeated = compute_match_score(required_skills, preferred_skills, repeated_eval, "Repeated", "repeated.pdf")

    assert score_single.required_supported == 1
    assert score_repeated.required_supported == 1
    assert score_single.final_score == score_repeated.final_score == 33.33



def test_suspicious_content_does_not_affect_score():
    """Verify: Suspicious content and prompt injection flags NEVER alter or penalize candidate match score."""
    required_skills = ["Python", "SQL"]
    preferred_skills = ["Docker"]

    evaluations = [
        SkillEvaluation(name="Python", category="required", status="supported", evidence="Python expert"),
        SkillEvaluation(name="SQL", category="required", status="missing", evidence=None),
        SkillEvaluation(name="Docker", category="preferred", status="supported", evidence="Docker compose"),
    ]

    # Clean candidate score
    clean_score = compute_match_score(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        evaluations=evaluations,
        candidate_name="Normal Candidate",
        filename="normal.pdf",
        is_suspicious=False,
        suspicious_findings=[],
    )

    # Adversarial candidate with identical skills but flagged with prompt injection
    flagged_score = compute_match_score(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        evaluations=evaluations,
        candidate_name="Adversarial Candidate",
        filename="adversarial.pdf",
        is_suspicious=True,
        suspicious_findings=[
            {"category": "Suspicious AI Instruction", "detected_text": "ignore instructions", "reason": "Test injection"}
        ],
    )

    # The score must be IDENTICAL (50% req * 0.8 + 100% pref * 0.2 = 40 + 20 = 60.0)
    assert clean_score.final_score == flagged_score.final_score == 60.0
    assert flagged_score.is_suspicious is True
    assert clean_score.is_suspicious is False


def test_tied_rankings_handled_fairly():
    """Verify: Equal scores receive identical ranks (ties) without artificial tie-breakers."""
    candidates = [
        CandidateScore(candidate_name="Candidate A", filename="a.pdf", status="Scored", final_score=90.0),
        CandidateScore(candidate_name="Candidate B", filename="b.pdf", status="Scored", final_score=85.0),
        CandidateScore(candidate_name="Candidate C", filename="c.pdf", status="Scored", final_score=85.0),
        CandidateScore(candidate_name="Candidate D", filename="d.pdf", status="Scored", final_score=70.0),
    ]

    ranked, needs_review = rank_and_group_candidates(candidates)

    assert len(ranked) == 4
    assert len(needs_review) == 0

    assert ranked[0].candidate_name == "Candidate A"
    assert ranked[0].rank == 1
    assert ranked[0].is_tie is False

    assert ranked[1].candidate_name in ["Candidate B", "Candidate C"]
    assert ranked[1].rank == 2
    assert ranked[1].is_tie is True
    assert "(Tie)" in ranked[1].display_rank

    assert ranked[2].candidate_name in ["Candidate B", "Candidate C"]
    assert ranked[2].rank == 2
    assert ranked[2].is_tie is True
    assert "(Tie)" in ranked[2].display_rank

    assert ranked[3].candidate_name == "Candidate D"
    assert ranked[3].rank == 4
    assert ranked[3].is_tie is False


def test_needs_review_candidates_excluded_from_ranking():
    """Verify: Candidates that could not be evaluated reliably are excluded from scored ranking."""
    candidates = [
        CandidateScore(candidate_name="Good Candidate", filename="good.pdf", status="Scored", final_score=95.0),
        CandidateScore(
            candidate_name="Corrupt PDF",
            filename="corrupt.pdf",
            status="Needs review",
            review_reason="Empty or unreadable PDF.",
            final_score=0.0,
        ),
    ]

    ranked, needs_review = rank_and_group_candidates(candidates)

    assert len(ranked) == 1
    assert ranked[0].candidate_name == "Good Candidate"
    assert ranked[0].rank == 1

    assert len(needs_review) == 1
    assert needs_review[0].candidate_name == "Corrupt PDF"
    assert needs_review[0].status == "Needs review"
