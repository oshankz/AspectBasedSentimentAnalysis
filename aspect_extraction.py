"""Aspect extraction logic using keyword mapping."""

from typing import Dict, List

# Predefined aspects and keyword dictionaries for rule-based aspect detection.
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "Faculty": ["faculty", "teacher", "professor", "lecturer", "mentor", "instructor"],
    "Infrastructure": ["infrastructure", "classroom", "lab", "library", "campus", "wifi", "facility"],
    "Curriculum": ["curriculum", "syllabus", "course", "subject", "module", "content"],
    "Placements": ["placement", "placements", "internship", "job", "career", "recruiter"],
    "Management": ["management", "administration", "office", "coordination", "staff", "support"],
}


def detect_aspects(text: str) -> List[str]:
    """Return all matching aspects found in feedback text."""
    text = (text or "").lower()
    matched_aspects = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            matched_aspects.append(aspect)

    # If no explicit aspect keyword is present, tag as Management by default.
    # This keeps output informative for generic comments.
    return matched_aspects if matched_aspects else ["Management"]
