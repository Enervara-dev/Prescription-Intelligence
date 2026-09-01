import re
from rapidfuzz import process, fuzz

# Frequency mapping
FREQ_MAP = {
    "OD": "Once Daily",
    "BD": "Twice Daily",
    "TDS": "Thrice Daily",
    "SOS": "When Required"
}

# Load medicine database once
def load_medicine_list(filepath="data/medicine_list.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

MEDICINE_DB = load_medicine_list()

# Common non-medicine words to ignore
COMMON_WORDS = {
    "name", "address", "tablet", "take", "with",
    "days", "day", "mg", "tab", "one", "two",
    "sig", "temp", "howvs"
}

def clean_line(line):
    line = re.sub(r'[^A-Za-z0-9\s/]', ' ', line)
    line = re.sub(r'\s+', ' ', line)
    return line.strip()

def is_medicine_candidate(word):
    if len(word) < 6:
        return False
    if not word.isalpha():
        return False
    if word.lower() in COMMON_WORDS:
        return False
    return True

def get_best_match(word):
    if not is_medicine_candidate(word):
        return None

    match, score, _ = process.extractOne(
        word,
        MEDICINE_DB,
        scorer=fuzz.partial_ratio
    )

    if score > 75:  # strict threshold
        return {"name": match, "confidence": score}

    return None

def extract_dosage(line):
    match = re.search(r'(\d{2,4})\s?(mg|ml|g)', line, re.IGNORECASE)
    return match.group(0) if match else "N/A"

def extract_duration(line):
    match = re.search(r'(\d+)\s?(day|days|d)\b', line, re.IGNORECASE)
    return match.group(0) if match else "N/A"

def extract_frequency(line):
    for key in FREQ_MAP:
        if re.search(r'\b' + key + r'\b', line.upper()):
            return FREQ_MAP[key]
    return "N/A"



def parse_medicines(text):
    lines = text.split("\n")
    results = []
    seen = set()
    
    for raw_line in lines:
        line = clean_line(raw_line)

        if len(line) < 5:
            continue

        words = line.split()

        for word in words:
            match = get_best_match(word)
            if match:
                results.append({
                    "name": match,
                    "dosage": extract_dosage(line),
                    "frequency": extract_frequency(line),
                    "duration": extract_duration(line)
                })
                break  # only one medicine per line

    return results