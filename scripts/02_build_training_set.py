import json
import random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = ROOT / "transcripts"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

records = []

n_samples = 800

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

samples = samples_positive[:int(n_samples/2)] + samples_negative[:int(n_samples/2)]
random.shuffle(samples)

with open(DATA_DIR / f"labeled_samples_{n_samples}.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

#with open('./data/negative_labels.json', "w", encoding="utf-8") as f:
#    json.dump(samples_negative, f, indent=2, ensure_ascii=False)

print(""" 
SUMMARY STATS 
Positive Labels: {0}
Negative Labels: {1}
""".format(len(samples_positive), len(samples_negative)))