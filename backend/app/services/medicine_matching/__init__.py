import os
from rapidfuzz import process, fuzz
from typing import Optional, Tuple
import re

# Load medicine database once
def load_medicine_list(filepath="app/data/medicine_list.txt"):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

MEDICINE_DB = load_medicine_list()

COMMON_WORDS = {
    "name", "address", "tablet", "take", "with",
    "days", "day", "mg", "tab", "one", "two",
    "sig", "temp", "howvs", "capsule", "cap", "syr",
    "syrup", "drop", "drops", "inj", "injection"
}

def clean_line(line: str) -> str:
    line = re.sub(r'[^A-Za-z0-9\s/]', ' ', line)
    line = re.sub(r'\s+', ' ', line)
    return line.strip()

def is_medicine_candidate(word: str) -> bool:
    if len(word) < 4:
        return False
    if word.lower() in COMMON_WORDS:
        return False
    return True

def find_medicine_match(raw_text: str) -> Tuple[Optional[str], float]:
    """
    Returns (best_match_name, confidence_score_0_to_1)
    """
    if not MEDICINE_DB:
        return None, 0.0
        
    cleaned = clean_line(raw_text)
    words = cleaned.split()
    
    best_match = None
    highest_score = 0.0
    
    for word in words:
        if not is_medicine_candidate(word):
            continue
            
        result = process.extractOne(
            word,
            MEDICINE_DB,
            scorer=fuzz.partial_ratio
        )
        
        if result:
            match_str, score, _ = result
            if score > highest_score:
                highest_score = score
                best_match = match_str
                
    # Normalize score from 0-100 to 0.0-1.0
    return best_match, highest_score / 100.0
