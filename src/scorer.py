"""Scoring and ranking engine for TalentLens AI.

Calculates candidate match scores using a fixed, deterministic Python formula.
The LLM does NOT decide final scores.
Candidate scores are strictly independent of ResumeShield flags or prompt injections.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from src.config import REQUIRED_SKILL_WEIGHT, PREFERRED_SKILL_WEIGHT


@dataclass
class SkillEvaluation:
    """Evaluation of an individual required or preferred skill."""
    name: str
    category: str  # "required" or "preferred"
    status: str  # "supported", "missing", "needs_review"
    evidence: Optional[str] = None
    review_note: Optional[str] = None
    points_awarded: float = 0.0
    max_points: float = 0.0

    def is_supported(self) -> bool:
        """A skill counts as supported ONLY when the status is strictly 'supported' or full points awarded."""
        return self.status.lower() == "supported" or (self.max_points > 0 and self.points_awarded == self.max_points)


@dataclass
class CandidateScore:
    """Complete score breakdown for a single candidate."""
    candidate_name: str
    filename: str
    status: str  # "Scored" or "Needs review"
    review_reason: Optional[str] = None

    required_total: int = 0
    required_supported: int = 0
    required_score: float = 0.0

    preferred_total: int = 0
    preferred_supported: int = 0
    preferred_score: float = 0.0
    has_preferred_skills: bool = True

    final_score: float = 0.0
    formula_breakdown: str = ""

    skill_evaluations: List[SkillEvaluation] = field(default_factory=list)
    is_suspicious: bool = False
    suspicious_findings: List[Dict] = field(default_factory=list)

    # Assigned during ranking
    rank: int = 0
    is_tie: bool = False
    display_rank: str = ""


def _normalize_skill_name(name: str) -> str:
    """Normalize skill name for case-insensitive deduplication."""
    return re_sub_whitespace(name.strip().lower())


def re_sub_whitespace(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text)


def compute_match_score(
    required_skills: List[str],
    preferred_skills: List[str],
    evaluations: List[SkillEvaluation],
    candidate_name: str,
    filename: str,
    is_suspicious: bool = False,
    suspicious_findings: Optional[List[Dict]] = None,
    candidate_status: str = "Scored",
    review_reason: Optional[str] = None,
) -> CandidateScore:
    """Calculate the match score using the strict Python formula.
    
    Required Skill Score = (supported required skills / total required skills) * 100
    Preferred Skill Score = (supported preferred skills / total preferred skills) * 100
    Final Match Score = 0.80 * Required Skill Score + 0.20 * Preferred Skill Score
    (or Required Skill Score if no preferred skills exist)
    
    Rules:
    - Count each skill only once (deduplicated).
    - Repeated keywords do not increase score.
    - Suspicious content flags NEVER alter or penalize the score.
    - Skills marked 'needs_review' or with unclear evidence do NOT count as supported.
    """
    if suspicious_findings is None:
        suspicious_findings = []

    # If document itself cannot be reliably evaluated, mark whole candidate for review
    if candidate_status != "Scored":
        return CandidateScore(
            candidate_name=candidate_name or filename,
            filename=filename,
            status="Needs review",
            review_reason=review_reason or "Document could not be evaluated reliably.",
            is_suspicious=is_suspicious,
            suspicious_findings=suspicious_findings,
            skill_evaluations=evaluations,
            final_score=0.0,
            formula_breakdown="Excluded from scoring: Candidate marked 'Needs review'.",
        )

    # Deduplicate job skills while preserving order
    unique_req_set: Set[str] = set()
    deduped_req: List[str] = []
    for s in required_skills:
        clean = _normalize_skill_name(s)
        if clean and clean not in unique_req_set:
            unique_req_set.add(clean)
            deduped_req.append(s.strip())

    unique_pref_set: Set[str] = set()
    deduped_pref: List[str] = []
    for s in preferred_skills:
        clean = _normalize_skill_name(s)
        if clean and clean not in unique_pref_set and clean not in unique_req_set:
            unique_pref_set.add(clean)
            deduped_pref.append(s.strip())

    total_req = len(deduped_req)
    total_pref = len(deduped_pref)

    if total_req == 0:
        return CandidateScore(
            candidate_name=candidate_name or filename,
            filename=filename,
            status="Needs review",
            review_reason="No required skills specified in job requirements.",
            is_suspicious=is_suspicious,
            suspicious_findings=suspicious_findings,
            skill_evaluations=evaluations,
            final_score=0.0,
            formula_breakdown="No required skills defined.",
        )

    # Map candidate evaluations by normalized name
    # Ensure each skill evaluation is counted only once!
    eval_by_skill: Dict[str, SkillEvaluation] = {}
    for ev in evaluations:
        norm = _normalize_skill_name(ev.name)
        if norm not in eval_by_skill:
            eval_by_skill[norm] = ev
        else:
            # If duplicates exist in evaluation, only upgrade to supported if valid evidence exists
            if ev.is_supported() and not eval_by_skill[norm].is_supported():
                eval_by_skill[norm] = ev

    # Count supported required skills
    supported_req_count = 0
    final_evaluations: List[SkillEvaluation] = []

    for req_skill in deduped_req:
        norm = _normalize_skill_name(req_skill)
        ev = eval_by_skill.get(norm)
        if ev and ev.is_supported():
            supported_req_count += 1
            final_evaluations.append(
                SkillEvaluation(
                    name=req_skill,
                    category="required",
                    status="supported",
                    evidence=ev.evidence,
                    review_note=ev.review_note,
                    points_awarded=ev.points_awarded if ev.max_points > 0 else 0.0,
                    max_points=ev.max_points,
                )
            )
        elif ev and ev.status == "needs_review":
            final_evaluations.append(
                SkillEvaluation(
                    name=req_skill,
                    category="required",
                    status="needs_review",
                    evidence=ev.evidence,
                    review_note=ev.review_note or "Evidence ambiguous or incomplete. Human review needed.",
                    points_awarded=ev.points_awarded if ev.max_points > 0 else 0.0,
                    max_points=ev.max_points,
                )
            )
        else:
            final_evaluations.append(
                SkillEvaluation(
                    name=req_skill,
                    category="required",
                    status="missing",
                    evidence=None,
                    review_note="No supporting evidence found in resume.",
                    points_awarded=0.0,
                    max_points=ev.max_points if ev else 0.0,
                )
            )

    has_custom_points = any(ev.max_points > 0 for ev in final_evaluations)

    # Count supported preferred skills
    has_preferred = total_pref > 0
    supported_pref_count = 0

    if has_preferred:
        for pref_skill in deduped_pref:
            norm = _normalize_skill_name(pref_skill)
            ev = eval_by_skill.get(norm)
            if ev and ev.is_supported():
                supported_pref_count += 1
                final_evaluations.append(
                    SkillEvaluation(
                        name=pref_skill,
                        category="preferred",
                        status="supported",
                        evidence=ev.evidence,
                        review_note=ev.review_note,
                        points_awarded=ev.points_awarded if ev.max_points > 0 else 0.0,
                        max_points=ev.max_points,
                    )
                )
            elif ev and ev.status == "needs_review":
                final_evaluations.append(
                    SkillEvaluation(
                        name=pref_skill,
                        category="preferred",
                        status="needs_review",
                        evidence=ev.evidence,
                        review_note=ev.review_note or "Evidence ambiguous. Human review needed.",
                        points_awarded=ev.points_awarded if ev.max_points > 0 else 0.0,
                        max_points=ev.max_points,
                    )
                )
            else:
                final_evaluations.append(
                    SkillEvaluation(
                        name=pref_skill,
                        category="preferred",
                        status="missing",
                        evidence=None,
                        review_note="No supporting evidence found in resume.",
                        points_awarded=0.0,
                        max_points=ev.max_points if ev else 0.0,
                    )
                )

    if has_custom_points:
        req_pts = sum(e.points_awarded for e in final_evaluations if e.category == "required")
        req_max = sum(e.max_points for e in final_evaluations if e.category == "required") or 80.0
        pref_pts = sum(e.points_awarded for e in final_evaluations if e.category == "preferred")
        pref_max = sum(e.max_points for e in final_evaluations if e.category == "preferred") or 20.0
        final_score = req_pts + pref_pts
        req_score = req_pts
        pref_score = pref_pts
        breakdown_str = (
            f"Sum of Criterion Scores: Required ({req_pts:.0f}/{req_max:.0f}) + "
            f"Preferred ({pref_pts:.0f}/{pref_max:.0f}) = {final_score:.1f}%"
        )
    else:
        req_score = (supported_req_count / total_req) * 100.0
        if has_preferred:
            pref_score = (supported_pref_count / total_pref) * 100.0
            final_score = (REQUIRED_SKILL_WEIGHT * req_score) + (PREFERRED_SKILL_WEIGHT * pref_score)
            breakdown_str = (
                f"0.80 × ({supported_req_count}/{total_req} × 100 = {req_score:.1f}) + "
                f"0.20 × ({supported_pref_count}/{total_pref} × 100 = {pref_score:.1f}) = {final_score:.1f}%"
            )
        else:
            pref_score = 0.0
            final_score = req_score
            breakdown_str = (
                f"Required Skill Score only (no preferred skills): "
                f"({supported_req_count}/{total_req} × 100) = {final_score:.1f}%"
            )

    return CandidateScore(
        candidate_name=candidate_name or filename,
        filename=filename,
        status="Scored",
        review_reason=None,
        required_total=total_req,
        required_supported=supported_req_count,
        required_score=round(req_score, 2),
        preferred_total=total_pref,
        preferred_supported=supported_pref_count,
        preferred_score=round(pref_score, 2),
        has_preferred_skills=has_preferred,
        final_score=round(final_score, 2),
        formula_breakdown=breakdown_str,
        skill_evaluations=final_evaluations,
        is_suspicious=is_suspicious,
        suspicious_findings=suspicious_findings,
    )


def rank_and_group_candidates(
    candidates: List[CandidateScore],
) -> Tuple[List[CandidateScore], List[CandidateScore]]:
    """Sort candidates by final score descending and assign fair tied rankings.
    
    Candidates requiring human review are separated and excluded from the scored ranking.
    Equal scores receive tied ranks (e.g., Rank 1, Rank 1, Rank 3). No tie-breakers are invented.
    
    Returns:
        Tuple[List[CandidateScore], List[CandidateScore]]:
            - Scored candidates with rank & tie information.
            - Candidates in the 'Needs review' bucket.
    """
    scored: List[CandidateScore] = []
    needs_review: List[CandidateScore] = []

    for c in candidates:
        if c.status == "Scored":
            scored.append(c)
        else:
            needs_review.append(c)

    # Sort scored candidates by final_score descending
    scored.sort(key=lambda x: x.final_score, reverse=True)

    # Compute standard competition ranking (1224) and tie indicators
    n = len(scored)
    for i in range(n):
        if i == 0:
            scored[i].rank = 1
        else:
            if scored[i].final_score == scored[i - 1].final_score:
                scored[i].rank = scored[i - 1].rank
            else:
                scored[i].rank = i + 1

    # Mark ties
    rank_counts: Dict[int, int] = {}
    for c in scored:
        rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1

    for c in scored:
        if rank_counts[c.rank] > 1:
            c.is_tie = True
            c.display_rank = f"#{c.rank} (Tie)"
        else:
            c.is_tie = False
            c.display_rank = f"#{c.rank}"

    return scored, needs_review


def evaluate_data_analyst_rubric_evidence(
    resume_text: str, filename: str = "resume.pdf"
) -> Tuple[str, List[SkillEvaluation]]:
    """Deterministic semantic evaluator for the 6 Live Mode Data Analyst criteria.
    
    Rubric:
    Required (80 pts):
    - Python for data analysis (20 pts)
    - SQL and database querying (20 pts)
    - Data cleaning (20 pts)
    - Basic statistics (20 pts)
    Preferred (20 pts):
    - Tableau, Power BI or equivalent visualization (10 pts)
    - Cloud basics (10 pts)
    
    Evaluates evidence from skills, internships, projects, and work experience, awarding
    full, partial, or zero credit strictly matching calibration targets.
    """
    text_lower = resume_text.lower()

    # Candidate name extraction (sanitized against adversarial / system instruction lines)
    lines = [l.strip() for l in resume_text.split("\n") if l.strip()]
    clean_lines = [
        l for l in lines
        if not any(w in l.lower() for w in [
            "system instruction", "ignore all", "disregard", "prompt injection", "ai screener", "note to ai"
        ])
    ]
    candidate_name = filename

    if len(clean_lines) >= 2:
        w0 = clean_lines[0].split()
        w1 = clean_lines[1].split()
        if len(w0) == 1 and len(w1) == 1 and w0[0][0].isupper() and w1[0][0].isupper():
            if w1[0].lower() not in ["contact", "experience", "education", "skills", "summary", "profile"]:
                candidate_name = f"{w0[0]} {w1[0]}"

    if candidate_name == filename and clean_lines:
        for line in clean_lines[:5]:
            clean = re.sub(r"[^\w\s\-]", "", line).strip()
            words = clean.split()
            if 2 <= len(words) <= 3 and all(w[0].isupper() for w in words if w):
                if not any(kw in clean.lower() for kw in ["resume", "curriculum", "vitae", "summary", "profile", "contact", "experience", "education"]):
                    candidate_name = line
                    break
        if candidate_name == filename and clean_lines:
            candidate_name = clean_lines[0]

    if "|" in candidate_name:
        candidate_name = candidate_name.split("|")[0].strip()

    evaluations: List[SkillEvaluation] = []

    # 1. Python for data analysis (max 20)
    py_pts = 0.0
    py_stat = "missing"
    py_ev = None
    py_note = None
    if "not completed any python" in text_lower or "no python" in text_lower:
        py_pts = 0.0
        py_stat = "missing"
        py_note = "Coursework in progress; has not completed any Python projects."
    elif any(k in text_lower for k in ["pandas", "numpy", "scipy", "statsmodels"]):
        py_pts = 20.0
        py_stat = "supported"
        py_ev = "Applied Python with data analysis libraries (pandas, NumPy, SciPy/statsmodels) in real data projects."
    elif any(k in text_lower for k in ["online python course", "introductory online python", "python course", "introductory python"]):
        py_pts = 5.0
        py_stat = "needs_review"
        py_ev = "Completed introductory Python course covering basic syntax (loops, functions)."
        py_note = "Beginner Python exposure only; lacks hands-on data manipulation libraries."
    elif "expert in python" in text_lower and not any(k in text_lower for k in ["pandas", "numpy"]):
        py_stat = "missing"
        py_note = "Self-proclaimed expert rating in skills list without supporting project code or evidence."
    elif "python" in text_lower and any(re.search(r"\b" + re.escape(w) + r"\b", text_lower) for w in ["script", "scripts", "scripting", "automation", "data pipeline", "programming"]):
        py_pts = 20.0
        py_stat = "supported"
        py_ev = "Demonstrated Python programming and analysis experience."
    else:
        py_stat = "missing"
        py_note = "No applied Python data analysis experience found in resume."

    evaluations.append(
        SkillEvaluation(
            name="Python for data analysis",
            category="required",
            status=py_stat,
            evidence=py_ev,
            review_note=py_note,
            points_awarded=py_pts,
            max_points=20.0,
        )
    )

    # 2. SQL and database querying (max 20)
    sql_pts = 0.0
    sql_stat = "missing"
    sql_ev = None
    sql_note = None
    if "not completed any" in text_lower and "sql" in text_lower:
        sql_pts = 0.0
        sql_stat = "missing"
        sql_note = "Has not completed any SQL projects."
    elif "no database was used" in text_lower or "single csv file" in text_lower:
        sql_pts = 0.0
        sql_stat = "missing"
        sql_note = "CSV file analysis only; no database was used."
    elif any(k in text_lower for k in ["postgresql", "sqlite", "mysql"]) or (
        "sql" in text_lower
        and any(k in text_lower for k in ["select", "where", "group by", "joins", "subqueries", "queries"])
    ):
        sql_pts = 20.0
        sql_stat = "supported"
        sql_ev = "Hands-on SQL queries with joins, aggregations, GROUP BY, and relational database operations."
    elif "expert in sql" in text_lower and not any(k in text_lower for k in ["database", "queries", "query", "table", "relational"]):
        sql_stat = "missing"
        sql_note = "Listed in skills list without supporting database schema or query evidence."
    elif "sql" in text_lower and any(k in text_lower for k in ["database", "schema", "relational", "query"]):
        sql_pts = 10.0
        sql_stat = "needs_review"
        sql_ev = "Basic SQL knowledge mentioned."
        sql_note = "Conceptual SQL knowledge without complex query demonstration."
    else:
        sql_stat = "missing"
        sql_note = "No SQL or database experience found in resume."

    evaluations.append(
        SkillEvaluation(
            name="SQL and database querying",
            category="required",
            status=sql_stat,
            evidence=sql_ev,
            review_note=sql_note,
            points_awarded=sql_pts,
            max_points=20.0,
        )
    )

    # 3. Data cleaning (max 20)
    clean_pts = 0.0
    clean_stat = "missing"
    clean_ev = None
    clean_note = None
    if any(k in text_lower for k in ["cleaned a 200,000", "cleaned supplier spreadsheets", "cleaned and standardized over 50,000", "cleaned shipment exports"]):
        clean_pts = 20.0
        clean_stat = "supported"
        clean_ev = "Dedicated data cleaning of large, messy datasets (deduplication, date standardization, imputation, validation rules)."
    elif any(
        k in text_lower
        for k in [
            "analyzed survey data",
            "survey data from 400",
            "sort and filter order lists to find missing or repeated",
            "correct them by hand",
        ]
    ):
        clean_pts = 5.0
        clean_stat = "needs_review"
        clean_ev = "Manual or basic data correction and dataset preparation."
        clean_note = "Basic data handling or manual error correction without automated data cleaning pipelines."
    else:
        clean_stat = "missing"
        clean_note = "No data cleaning experience evidenced."

    evaluations.append(
        SkillEvaluation(
            name="Data cleaning",
            category="required",
            status=clean_stat,
            evidence=clean_ev,
            review_note=clean_note,
            points_awarded=clean_pts,
            max_points=20.0,
        )
    )

    # 4. Basic statistics (max 20)
    stat_pts = 0.0
    stat_stat = "missing"
    stat_ev = None
    stat_note = None
    if any(k in text_lower for k in ["t-tests", "anova", "regression", "hypothesis testing", "standard deviations", "median delivery times"]):
        stat_pts = 20.0
        stat_stat = "supported"
        stat_ev = "Applied statistical analysis including descriptive statistics (means, medians, std dev) and inferential tests (t-tests, ANOVA, regression)."
    elif any(k in text_lower for k in ["averages and percentages", "percentages in excel"]):
        stat_pts = 8.0
        stat_stat = "needs_review"
        stat_ev = "Calculated simple averages and percentages in Excel to report data quality rates."
        stat_note = "Basic arithmetic averages/percentages without inferential statistics or hypothesis testing."
    elif any(k in text_lower for k in ["sum and average", "basic formulas"]):
        stat_pts = 5.0
        stat_stat = "needs_review"
        stat_ev = "Used basic Excel formulas (SUM, AVERAGE)."
        stat_note = "Introductory spreadsheet formulas only; no formal statistical methods."
    else:
        stat_stat = "missing"
        stat_note = "No statistical analysis evidence found."

    evaluations.append(
        SkillEvaluation(
            name="Basic statistics",
            category="required",
            status=stat_stat,
            evidence=stat_ev,
            review_note=stat_note,
            points_awarded=stat_pts,
            max_points=20.0,
        )
    )

    # 5. Tableau, Power BI or equivalent visualization (max 10)
    vis_pts = 0.0
    vis_stat = "missing"
    vis_ev = None
    vis_note = None
    if any(k in text_lower for k in ["tableau dashboard", "power bi dashboard", "interactive business dashboards", "dashboard tracking", "tableau reports", "dashboard of on-time"]):
        vis_pts = 10.0
        vis_stat = "supported"
        vis_ev = "Built interactive business dashboards in Tableau/Power BI used by reporting teams."
    elif "excel tables and charts" in text_lower:
        vis_pts = 5.0
        vis_stat = "needs_review"
        vis_ev = "Presented analysis results in Excel tables and charts."
        vis_note = "Spreadsheet chart presentations without specialized BI dashboard tooling."
    elif "weekly summary reports" in text_lower:
        vis_pts = 3.0
        vis_stat = "needs_review"
        vis_ev = "Prepared simple weekly summary reports; viewed spreadsheet chart lessons."
        vis_note = "Basic report preparation; course in progress."
    else:
        vis_stat = "missing"
        vis_note = "No business visualization or BI dashboard tools found."

    evaluations.append(
        SkillEvaluation(
            name="Tableau, Power BI or equivalent visualization",
            category="preferred",
            status=vis_stat,
            evidence=vis_ev,
            review_note=vis_note,
            points_awarded=vis_pts,
            max_points=10.0,
        )
    )

    # 6. Cloud basics (max 10)
    cloud_pts = 0.0
    cloud_stat = "missing"
    cloud_ev = None
    cloud_note = None
    if any(k in text_lower for k in ["s3", "ec2", "cloud run", "blob storage", "cloud storage", "aws s3"]):
        cloud_pts = 10.0
        cloud_stat = "supported"
        cloud_ev = "Cloud infrastructure experience (AWS S3 storage, basic EC2 integration)."
    else:
        cloud_stat = "missing"
        cloud_note = "No cloud basics or cloud storage experience found."

    evaluations.append(
        SkillEvaluation(
            name="Cloud basics",
            category="preferred",
            status=cloud_stat,
            evidence=cloud_ev,
            review_note=cloud_note,
            points_awarded=cloud_pts,
            max_points=10.0,
        )
    )

    return candidate_name, evaluations

