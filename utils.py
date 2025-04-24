import re
import spacy
from typing import List, Dict, Tuple

# Load spaCy model for NER (use small English model)
nlp = spacy.load("en_core_web_sm")

# Regex patterns for PII/PCI entities
PATTERNS = {
    "full_name": r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)",  # Simple pattern for full names
    "email": r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
    "phone_number": r"(\+?\d{1,3}[-.\s]??\d{1,4}[-.\s]??\d{3,4}[-.\s]??\d{3,4})",
    "dob": r"(\b(?:0?[1-9]|[12][0-9]|3[01])[-/](?:0?[1-9]|1[012])[-/](?:19|20)\d{2}\b)",  # dd/mm/yyyy or dd-mm-yyyy
    "aadhar_num": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "credit_debit_no": r"\b(?:\d[ -]*?){13,16}\b",
    "cvv_no": r"\b\d{3,4}\b",
    "expiry_no": r"\b(0[1-9]|1[0-2])\/?([0-9]{2}|[0-9]{4})\b"
}

def mask_pii(text: str) -> Tuple[str, List[Dict]]:
    """
    Mask PII and PCI entities in the text.
    Returns masked text and list of masked entities with positions, classification, and original entity.
    """
    masked_entities = []
    masked_text = text

    # To avoid overlapping replacements, keep track of offsets
    offset = 0

    # Find all matches for each pattern
    matches = []
    for entity, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            matches.append((start, end, entity, match.group()))

    # Sort matches by start position
    matches = sorted(matches, key=lambda x: x[0])

    # Remove overlapping matches by keeping the longest match
    filtered_matches = []
    prev_end = -1
    for m in matches:
        if m[0] >= prev_end:
            filtered_matches.append(m)
            prev_end = m[1]

    # Replace matches with masked tags from the end to avoid messing up indices
    for start, end, entity, original in reversed(filtered_matches):
        masked_entities.append({
            "position": [start, end],
            "classification": entity,
            "entity": original
        })
        masked_text = masked_text[:start] + f"[{entity}]" + masked_text[end:]

    # Reverse masked_entities to be in order of appearance
    masked_entities.reverse()

    return masked_text, masked_entities

def demask_pii(masked_text: str, masked_entities: List[Dict]) -> str:
    """
    Restore the original PII entities in the masked text.
    """
    demasked_text = masked_text
    # Replace masked tags with original entities from the end to avoid messing indices
    for entity_info in reversed(masked_entities):
        start, end = entity_info["position"]
        classification = entity_info["classification"]
        original = entity_info["entity"]
        tag = f"[{classification}]"
        # Find the tag in demasked_text starting from start position
        tag_pos = demasked_text.find(tag)
        if tag_pos != -1:
            demasked_text = demasked_text[:tag_pos] + original + demasked_text[tag_pos + len(tag):]

    return demasked_text
