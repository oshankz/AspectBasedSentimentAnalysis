"""
aspect_extraction.py
--------------------
Identifies aspects mentioned in student feedback using keyword mapping.
Predefined aspects: Faculty, Infrastructure, Curriculum, Placements, Management.
"""

# Keyword dictionary mapping aspect names to related keywords
ASPECT_KEYWORDS = {
    "Faculty": [
        "teacher", "professor", "faculty", "instructor", "lecturer", "mentor",
        "teaching", "taught", "explain", "explanation", "class", "lecture",
        "doubt", "staff", "educator", "guide", "tutor", "academic", "knowledgeable",
        "knowledge", "subject", "lesson", "session", "approach", "method"
    ],
    "Infrastructure": [
        "infrastructure", "building", "lab", "laboratory", "library", "hostel",
        "canteen", "cafeteria", "classroom", "facility", "facilities", "campus",
        "wifi", "internet", "computer", "equipment", "cleanliness", "clean",
        "maintenance", "parking", "sports", "gym", "auditorium", "room",
        "bench", "projector", "ac", "air conditioning", "toilet", "washroom"
    ],
    "Curriculum": [
        "curriculum", "syllabus", "course", "subject", "study", "material",
        "content", "module", "assignment", "project", "exam", "examination",
        "test", "practical", "theory", "research", "learning", "academic",
        "schedule", "timetable", "workshop", "seminar", "program", "degree",
        "skill", "knowledge", "internship", "training"
    ],
    "Placements": [
        "placement", "job", "company", "recruit", "recruitment", "hire",
        "hiring", "career", "offer", "salary", "package", "interview",
        "campus", "drive", "opportunity", "employ", "employment", "industry",
        "corporate", "mnc", "startup", "internship", "profile", "lpa", "ctc"
    ],
    "Management": [
        "management", "admin", "administration", "principal", "director",
        "hod", "dean", "college", "department", "rule", "policy", "fee",
        "fees", "regulation", "event", "fest", "activity", "committee",
        "response", "support", "complaint", "grievance", "discipline",
        "organization", "coordination", "communication"
    ]
}


def extract_aspects(text: str) -> list:
    """
    Identify which aspects are mentioned in the given feedback text.

    Parameters:
        text (str): Raw or preprocessed feedback string.

    Returns:
        list: List of detected aspect names. Returns ['General'] if none found.
    """
    text_lower = text.lower()
    detected = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            # Match whole words to avoid false positives
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                if aspect not in detected:
                    detected.append(aspect)
                break  # One match per aspect is enough

    return detected if detected else ["General"]


def get_aspect_keywords(aspect: str) -> list:
    """Return the keyword list for a given aspect."""
    return ASPECT_KEYWORDS.get(aspect, [])


def get_all_aspects() -> list:
    """Return all predefined aspect names."""
    return list(ASPECT_KEYWORDS.keys())


# Import re here since it's used inside extract_aspects
import re
