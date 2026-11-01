"""Feedback generation module for TalentLens AI.

Constructs comprehensive individual candidate reports including:
- Score breakdowns & formula explanation
- Verifiable evidence for supported skills
- Missing skill gaps & actionable, constructive improvement suggestions
- Items flagged for human review
- Completely separate ResumeShield suspicious-content audit reports
"""

from __future__ import annotations
from typing import Dict, Any, List
from src.scorer import CandidateScore


# Library of constructive skill improvement recommendations
SKILL_SUGGESTION_TEMPLATES: Dict[str, str] = {
    "python": "Develop an end-to-end Python project showcasing clean code, type hinting, and unit test coverage on GitHub.",
    "pymupdf": "Build a document processing utility utilizing PyMuPDF (fitz) for text extraction, layout inspection, and metadata parsing.",
    "pdf processing": "Explore libraries like PyMuPDF or pypdf to implement automated PDF parsing and document information retrieval pipelines.",
    "rest apis": "Design and document RESTful microservices using FastAPI or Flask, including OpenAPI/Swagger documentation and JWT authentication.",
    "sql": "Practice advanced relational queries, multi-table joins, indexing optimization, and schema design in PostgreSQL or MySQL.",
    "pytest": "Implement a full automated testing suite with pytest, including unit tests, fixtures, parameterization, and mock integrations.",
    "automated testing": "Add CI/CD pipeline automation (e.g., GitHub Actions) executing automated unit and integration tests on pull requests.",
    "docker": "Containerize backend services with Docker and Docker Compose, optimizing multi-stage builds and minimal runtime images.",
    "streamlit": "Build and deploy an interactive data dashboard or prototype using Streamlit and share the live app on Streamlit Community Cloud.",
    "cloud services": "Deploy a project onto Google Cloud Run, AWS ECS, or Azure App Service to demonstrate cloud architecture fundamentals.",
    "git": "Demonstrate collaborative Git workflows with branching strategies, clear pull request descriptions, and semantic commit history.",
    "machine learning": "Build a hands-on ML model with scikit-learn or PyTorch, documenting feature engineering, validation metrics, and inference benchmarks.",
    "fastapi": "Create high-performance asynchronous API endpoints in FastAPI with Pydantic schema validation.",
}


def _get_constructive_suggestion(skill_name: str) -> str:
    """Generate a tailored, constructive suggestion for a missing skill."""
    lower = skill_name.strip().lower()
    for key, suggestion in SKILL_SUGGESTION_TEMPLATES.items():
        if key in lower:
            return suggestion

    return (
        f"Gain hands-on experience in '{skill_name}' by building a focused portfolio project, "
        "completing targeted coursework, or contributing to relevant open-source repositories."
    )


def generate_candidate_feedback(candidate: CandidateScore) -> Dict[str, Any]:
    """Generate a structured, user-facing feedback report for a candidate."""
    supported_skills = []
    missing_skills = []
    review_items = []

    for ev in candidate.skill_evaluations:
        if ev.status == "supported":
            supported_skills.append({
                "skill": ev.name,
                "category": ev.category.capitalize(),
                "evidence": ev.evidence or "Mentioned with direct contextual support.",
            })
        elif ev.status == "needs_review":
            review_items.append({
                "skill": ev.name,
                "category": ev.category.capitalize(),
                "review_note": ev.review_note or "Evidence ambiguous or incomplete; manual human check required.",
                "evidence": ev.evidence,
            })
        else:
            missing_skills.append({
                "skill": ev.name,
                "category": ev.category.capitalize(),
            })

    # Suggestions for missing skills
    suggestions: List[Dict[str, str]] = []
    for m in missing_skills:
        suggestions.append({
            "skill": m["skill"],
            "category": m["category"],
            "suggestion": _get_constructive_suggestion(m["skill"]),
        })

    # If document level review is needed
    if candidate.status == "Needs review" and candidate.review_reason:
        review_items.insert(0, {
            "skill": "Document Readability",
            "category": "Document",
            "review_note": candidate.review_reason,
            "evidence": None,
        })

    return {
        "candidate_name": candidate.candidate_name,
        "filename": candidate.filename,
        "status": candidate.status,
        "rank": candidate.display_rank,
        "final_score": candidate.final_score,
        "formula_breakdown": candidate.formula_breakdown,
        "required_summary": {
            "supported": candidate.required_supported,
            "total": candidate.required_total,
            "score": candidate.required_score,
        },
        "preferred_summary": {
            "has_preferred": candidate.has_preferred_skills,
            "supported": candidate.preferred_supported,
            "total": candidate.preferred_total,
            "score": candidate.preferred_score,
        },
        "supported_skills": supported_skills,
        "missing_skills": missing_skills,
        "review_items": review_items,
        "suggestions": suggestions,
        "suspicious_report": {
            "is_flagged": candidate.is_suspicious,
            "findings_count": len(candidate.suspicious_findings),
            "findings": candidate.suspicious_findings,
            "disclaimer": (
                "ResumeShield heuristics inspect observable text attributes and common prompt injection patterns. "
                "Detection is imperfect and may produce false positives or false negatives. Flags do NOT automatically "
                "penalize scores or disqualify candidates. All final decisions must be made by human reviewers."
            ),
        },
    }
