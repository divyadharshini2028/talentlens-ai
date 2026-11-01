"""ResumeShield: PDF text extraction and forensics module.

Inspects PDF documents for extractable text, readability, and observable indicators of
hidden text or prompt injection attempts (e.g., white text, micro-fonts, off-page coordinates,
overlapping text, and AI-targeting instructions).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import pymupdf

from src.config import (
    MIN_EXTRACTABLE_TEXT_CHARS,
    MICRO_FONT_SIZE_THRESHOLD,
    WHITE_COLOR_THRESHOLD,
)


@dataclass
class SuspiciousFinding:
    """Represents an observable suspicious indicator detected in a PDF."""
    category: str
    detected_text: str
    reason: str
    severity: str  # "info", "warning", "high"
    page: int
    detection_type: str = ""  # Specific classification: "Hidden white text", "Very small-font instructions", "AI-targeted prompt injection", "Instructions to manipulate candidate ranking", "Suspicious claims without supporting evidence", etc.


@dataclass
class InspectionResult:
    """Outcome of PDF extraction and ResumeShield forensics."""
    filename: str
    status: str  # "OK" or "Needs review"
    review_reason: Optional[str] = None
    is_suspicious: bool = False
    findings: List[SuspiciousFinding] = field(default_factory=list)
    raw_text: str = ""
    page_count: int = 0
    char_count: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "status": self.status,
            "review_reason": self.review_reason,
            "is_suspicious": self.is_suspicious,
            "findings": [
                {
                    "category": f.category,
                    "detection_type": f.detection_type or f.category,
                    "detected_text": f.detected_text,
                    "reason": f.reason,
                    "severity": f.severity,
                    "page": f.page,
                }
                for f in self.findings
            ],
            "raw_text": self.raw_text,
            "page_count": self.page_count,
            "char_count": self.char_count,
            "error": self.error,
        }


# Regular expressions for suspicious AI-targeting instructions / prompt injections
# Format: (pattern, detection_type, reason, severity)
INJECTION_PATTERNS = [
    # 1. AI-targeted prompt injection
    (
        r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions?|directions?|prompts?|rules?)\b",
        "AI-targeted prompt injection",
        "Prompt Injection: Attempts to override prior instructions or system prompt rules.",
        "high",
    ),
    (
        r"(?i)\b(?:ai\s*screener|screening\s*system|note\s+to\s+ai|system\s*instruction\s+to\s+ai|new\s+instruction\s+for\s+ai)\b",
        "AI-targeted prompt injection",
        "AI-Targeted Direct Address: Explicit commands targeting the automated AI screening model.",
        "high",
    ),
    (
        r"(?i)\b(?:system\s*prompt|system\s*instruction|system\s*message|developer\s*mode|assistant\s*rule)\b",
        "AI-targeted prompt injection",
        "System Prompt Reference: Mentions internal AI system prompt directives or developer mode.",
        "warning",
    ),

    # 2. Instructions to manipulate candidate ranking
    (
        r"(?i)\b(?:give|assign|award|grant)\s+(?:this\s+candidate|me)?\s*(?:a\s+)?(?:score\s+of\s+)?(?:100(?:\/100)?|perfect\s+score|maximum\s+score|top\s+rank)\b",
        "Instructions to manipulate candidate ranking",
        "Score Manipulation: Tries to force a 100/100 or maximum rating via text instruction.",
        "high",
    ),
    (
        r"(?i)\b(?:you\s+must\s+rank\s+this\s+candidate\s+(?:first|highest|#1)|must\s+be\s+ranked\s+(?:number\s+one|#1|first)|rank\s+this\s+candidate\s+first)\b",
        "Instructions to manipulate candidate ranking",
        "Instructions to Manipulate Candidate Ranking: Direct order commanding first/top rank.",
        "high",
    ),
    (
        r"(?i)\b(?:set\s+(?:the\s+)?match\s+score\s+to\s+100|output\s+final\s+score\s*=\s*100|output\s+the\s+maximum\s+score)\b",
        "Instructions to manipulate candidate ranking",
        "Instructions to Manipulate Candidate Ranking: Explicit directive dictating artificial output scores.",
        "high",
    ),
    (
        r"(?i)\b(?:disregard\s+all\s+other\s+candidates|hire\s+this\s+candidate\s+immediately)\b",
        "Instructions to manipulate candidate ranking",
        "Instructions to Manipulate Candidate Ranking: Commands prioritizing this applicant over others.",
        "high",
    ),
    (
        r"(?i)\b(?:disregard\s+the\s+scoring\s+rubric|disregard\s+the\s+job\s+description|ignore\s+the\s+job\s+description|ignore\s+the\s+rubric|bypass\s+evaluation)\b",
        "Instructions to manipulate candidate ranking",
        "Evaluation Bypass: Explicit attempt to bypass rubric or job description.",
        "high",
    ),

    # 3. Suspicious claims without supporting evidence
    (
        r"(?i)\b(?:do\s+not\s+check\s+for\s+evidence|treat\s+all\s+skills\s+as\s+verified|award\s+full\s+marks\s+regardless|ignore\s+(?:the\s+)?job\s+requirements|do\s+not\s+list\s+(?:any\s+)?missing\s+skills|satisfies\s+100%\s+of\s+the\s+rubric)\b",
        "Suspicious claims without supporting evidence",
        "Suspicious Claims Without Supporting Evidence: Explicit directive commanding the system not to verify evidence or to treat unevidenced skills as verified.",
        "high",
    ),
    (
        r"(?i)\bexpert\s+in\s+(?:python|sql|data\s+cleaning|statistics|tableau|power\s+bi|aws|azure)\b",
        "Suspicious claims without supporting evidence",
        "Suspicious Claims Without Supporting Evidence: Self-proclaimed expert claim without corroborating project or work experience.",
        "warning",
    ),

    # 4. Hidden Markup Comments
    (
        r"<!--[\s\S]*?-->|\[//\]:\s*#\s*\([^\)]*\)",
        "AI-targeted prompt injection",
        "Hidden Markup Comments: HTML/Markdown comments embedded inside text.",
        "info",
    ),
]


def _srgb_to_rgb(color_int: int) -> Tuple[int, int, int]:
    """Convert an integer sRGB color representation into (R, G, B) tuple."""
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return r, g, b


def _check_prompt_injections(text: str, page_num: int) -> List[SuspiciousFinding]:
    """Scan text for common adversarial instructions and prompt injection attempts."""
    findings: List[SuspiciousFinding] = []
    for item in INJECTION_PATTERNS:
        if len(item) == 4:
            pattern, detection_type, description, severity = item
        else:
            pattern, description, severity = item
            detection_type = "AI-targeted prompt injection"

        matches = re.finditer(pattern, text)
        for match in matches:
            matched_snippet = match.group(0).strip()
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].replace("\n", " ").strip()

            findings.append(
                SuspiciousFinding(
                    category="Suspicious AI Instruction",
                    detection_type=detection_type,
                    detected_text=f'"{context}"',
                    reason=f"{description} (Matched: '{matched_snippet}')",
                    severity=severity,
                    page=page_num,
                )
            )
    return findings


def inspect_pdf_bytes(pdf_bytes: bytes, filename: str = "uploaded.pdf") -> InspectionResult:
    """Inspect a PDF provided as raw bytes.
    
    Extracts text, identifies unreadable or empty documents, and executes
    ResumeShield forensic checks on font colors, sizes, positions, and instruction text.
    """
    if not pdf_bytes:
        return InspectionResult(
            filename=filename,
            status="Needs review",
            review_reason="Empty file uploaded (0 bytes received).",
            error="Empty file",
        )

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return InspectionResult(
            filename=filename,
            status="Needs review",
            review_reason=f"Malformed or corrupted PDF file. Could not parse document: {str(e)}",
            error=str(e),
        )

    try:
        page_count = len(doc)
        if page_count == 0:
            doc.close()
            return InspectionResult(
                filename=filename,
                status="Needs review",
                review_reason="PDF document has 0 pages.",
                page_count=0,
            )

        extracted_text_pieces: List[str] = []
        findings: List[SuspiciousFinding] = []
        seen_span_boxes: List[Tuple[int, float, float, float, float, str]] = []

        for page_idx in range(page_count):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_rect = page.rect
            page_text = page.get_text()
            extracted_text_pieces.append(page_text)

            # Check prompt injection patterns in plain extracted page text
            page_injections = _check_prompt_injections(page_text, page_num)
            findings.extend(page_injections)

            # Deep forensics: Expand extraction clip box to catch off-page / cloaked text
            try:
                expanded_clip = pymupdf.Rect(
                    page_rect.x0 - 2000,
                    page_rect.y0 - 2000,
                    page_rect.x1 + 2000,
                    page_rect.y1 + 2000,
                )
                page_dict = page.get_text("dict", clip=expanded_clip)
                blocks = page_dict.get("blocks", [])
                for block in blocks:
                    if block.get("type") != 0:  # 0 indicates text block
                        continue

                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if not text:
                                continue

                            font_size = float(span.get("size", 12.0))
                            color_val = span.get("color", 0)
                            bbox = span.get("bbox", (0, 0, 0, 0))  # (x0, y0, x1, y1)

                            r, g, b = _srgb_to_rgb(color_val)
                            x0, y0, x1, y1 = bbox

                            # 1. Off-page Positioning: Coordinates outside visible page bounding box
                            if (
                                x1 < page_rect.x0 - 3
                                or y1 < page_rect.y0 - 3
                                or x0 > page_rect.x1 + 3
                                or y0 > page_rect.y1 + 3
                            ):
                                findings.append(
                                    SuspiciousFinding(
                                        category="Off-Page Text",
                                        detection_type="Off-page text coordinates",
                                        detected_text=text[:120],
                                        reason=(
                                            f"Text coordinates [{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}] "
                                            f"fall outside visible page boundaries [{page_rect.width:.1f}x{page_rect.height:.1f}]."
                                        ),
                                        severity="high",
                                        page=page_num,
                                    )
                                )
                                # Check if off-page text has prompt injection
                                off_injections = _check_prompt_injections(text, page_num)
                                findings.extend(off_injections)

                            # 2. Unusual Text Color: White / Near-White text
                            if (
                                r >= WHITE_COLOR_THRESHOLD
                                and g >= WHITE_COLOR_THRESHOLD
                                and b >= WHITE_COLOR_THRESHOLD
                            ):
                                findings.append(
                                    SuspiciousFinding(
                                        category="Hidden White Text",
                                        detection_type="Hidden white text",
                                        detected_text=text[:120],
                                        reason=(
                                            f"Text color is white/near-white RGB({r},{g},{b}), "
                                            f"which is virtually invisible against standard white page backgrounds."
                                        ),
                                        severity="high",
                                        page=page_num,
                                    )
                                )
                                # Check if hidden white text contains injection
                                white_injections = _check_prompt_injections(text, page_num)
                                findings.extend(white_injections)

                            # 3. Unusual Text Size: Micro-text (< 3pt)
                            if font_size < MICRO_FONT_SIZE_THRESHOLD and len(text) > 2:
                                findings.append(
                                    SuspiciousFinding(
                                        category="Micro-Font Text",
                                        detection_type="Very small-font instructions",
                                        detected_text=text[:120],
                                        reason=(
                                            f"Extremely small font size ({font_size:.1f}pt < "
                                            f"{MICRO_FONT_SIZE_THRESHOLD}pt threshold), often used to hide keywords or instructions."
                                        ),
                                        severity="warning",
                                        page=page_num,
                                    )
                                )

                            # 4. Overlapping Text Detection: Identical bounding box with different text
                            if len(text) > 3:
                                for seen_p, sx0, sy0, sx1, sy1, seen_t in seen_span_boxes:
                                    if seen_p == page_num:
                                        if (
                                            abs(x0 - sx0) < 1.0
                                            and abs(y0 - sy0) < 1.0
                                            and abs(x1 - sx1) < 1.0
                                            and abs(y1 - sy1) < 1.0
                                            and text != seen_t
                                        ):
                                            findings.append(
                                                SuspiciousFinding(
                                                    category="Overlapping Text",
                                                    detection_type="Overlapping text",
                                                    detected_text=f"'{text}' over '{seen_t}'",
                                                    reason="Multiple text strings placed directly on top of each other.",
                                                    severity="warning",
                                                    page=page_num,
                                                )
                                            )
                                            break
                                seen_span_boxes.append((page_num, x0, y0, x1, y1, text))

            except Exception as forensics_err:
                findings.append(
                    SuspiciousFinding(
                        category="Forensic Inspection Warning",
                        detected_text="",
                        reason=f"Partial PDF forensics error on page {page_num}: {str(forensics_err)}",
                        severity="info",
                        page=page_num,
                    )
                )

        full_raw_text = "\n\n".join(extracted_text_pieces).strip()
        char_count = len(full_raw_text)

        # De-duplicate identical findings on same page
        unique_findings: List[SuspiciousFinding] = []
        seen_keys = set()
        for f in findings:
            key = (f.category, f.page, f.detected_text.strip()[:60])
            if key not in seen_keys:
                seen_keys.add(key)
                unique_findings.append(f)

        # Check for unreadable / scanned / empty content
        if char_count < MIN_EXTRACTABLE_TEXT_CHARS:
            doc.close()
            return InspectionResult(
                filename=filename,
                status="Needs review",
                review_reason=(
                    f"Document contains insufficient extractable text ({char_count} characters). "
                    "The file appears to be empty, scanned/image-based (requiring OCR), or encrypted."
                ),
                is_suspicious=len(unique_findings) > 0,
                findings=unique_findings,
                raw_text=full_raw_text,
                page_count=page_count,
                char_count=char_count,
            )

        doc.close()
        return InspectionResult(
            filename=filename,
            status="OK",
            review_reason=None,
            is_suspicious=len(unique_findings) > 0,
            findings=unique_findings,
            raw_text=full_raw_text,
            page_count=page_count,
            char_count=char_count,
        )

    except Exception as e:
        try:
            doc.close()
        except Exception:
            pass
        return InspectionResult(
            filename=filename,
            status="Needs review",
            review_reason=f"Unexpected error while extracting PDF: {str(e)}",
            error=str(e),
        )
