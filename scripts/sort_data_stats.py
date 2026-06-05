import json
import random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = ROOT / "transcripts"

records = []

for file_path in TRANSCRIPTS_DIR.glob("*.json"):
    try:
        record = {}
        with open(file_path, 'r', encoding='utf-8') as f: 
            record = json.load(f)

        records.append(record)

    except Exception as e: 
        print(e) 

samples_positive = [] 
samples_negative = [] 

for record in records: 
    for sample in record: 
        if sample['label'] == 1: 
            samples_positive.append(sample)
        else: 
            samples_negative.append(sample) 

random.shuffle(samples_positive)
random.shuffle(samples_negative)

#with open('./data/positive_labels.json', "w", encoding="utf-8") as f:
#    json.dump(samples_positive, f, indent=2, ensure_ascii=False)

#with open('./data/negative_labels.json', "w", encoding="utf-8") as f:
#    json.dump(samples_negative, f, indent=2, ensure_ascii=False)

print(""" 
SUMMARY STATS 
Positive Labels: {0}
Negative Labels: {1}
""".format(len(samples_positive), len(samples_negative)))