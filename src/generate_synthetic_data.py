"""
Generate simple synthetic medical records in JSONL format.
Each record is a short "clinical note" with a patient id and a note text.
"""

import json
import random
import argparse
import os
from tqdm import tqdm

SYMPTOMS = [
    "fever", "cough", "headache", "nausea", "fatigue", "shortness of breath",
    "chest pain", "abdominal pain", "dizziness", "sore throat"
]

DIAGNOSES = [
    "viral infection", "bacterial infection", "migraine", "gastroenteritis",
    "asthma", "hypertension", "myocardial ischemia", "appendicitis"
]

TREATMENTS = [
    "rest and fluids", "antibiotics", "analgesics", "inhaler", "surgery",
    "IV fluids", "observation"
]

def make_record(i):
    # Compose a brief clinical note template
    n_symptoms = random.choice([1,2,3])
    symptoms = random.sample(SYMPTOMS, n_symptoms)
    diagnosis = random.choice(DIAGNOSES)
    treatment = random.choice(TREATMENTS)
    age = random.randint(10, 90)
    gender = random.choice(["male", "female", "non-binary"])
    note = (
        f"Patient {i}, {age}-year-old {gender}, presents with "
        f"{', '.join(symptoms)}. History suggests {diagnosis}. "
        f"Recommended: {treatment}."
    )
    return {
        "id": f"patient_{i}",
        "note": note,
        "age": age,
        "gender": gender,
        "diagnosis": diagnosis
    }

def split_and_write(records, out_dir):
    # 70% train, 30% test; also produce labels for membership experiments
    random.shuffle(records)
    n = len(records)
    split = int(0.7 * n)
    train = records[:split]
    test = records[split:]
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/synthetic_train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(f"{out_dir}/synthetic_test.jsonl", "w") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(train)} train and {len(test)} test records to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_records", type=int, default=2000)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    records = [make_record(i) for i in range(args.n_records)]
    split_and_write(records, args.out_dir)
