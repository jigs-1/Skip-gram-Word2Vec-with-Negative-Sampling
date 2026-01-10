import re

INPUT_PATH = "data\\enwik8\\enwik8"
OUTPUT_PATH = "data\\wiki_clean.txt"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)        # remove markup
    text = re.sub(r"[^a-z ]+", " ", text)     # keep letters only
    text = re.sub(r"\s+", " ", text)
    return text.strip()

with open(INPUT_PATH, "r", encoding="utf-8", errors="ignore") as f:
    raw_text = f.read()

cleaned = clean_text(raw_text)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(cleaned)

print("Preprocessing complete. Saved to wiki_clean.txt")