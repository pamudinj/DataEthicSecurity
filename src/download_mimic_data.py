"""
Download and prepare MIMIC-III dataset for membership inference testing.
MIMIC-III is a publicly available de-identified critical care dataset.

Reference: Johnson et al. (2016) MIMIC-III, a freely accessible critical care database.
Scientific Data 3:160035

Requirements:
1. PhysioNet account (free): https://physionet.org/
2. Agree to MIMIC-III data use agreement
3. Run: python -m pip install --upgrade pip && pip install wfdb
"""

import argparse
import json
import os
from pathlib import Path
import sys

class MIMICDataPreparation:
    """Handles MIMIC-III data download and preparation."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def print_setup_instructions(self):
        """Print instructions for accessing MIMIC-III."""
        instructions = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    MIMIC-III Dataset Setup Instructions                    ║
╚════════════════════════════════════════════════════════════════════════════╝

ABOUT MIMIC-III:
  - Largest publicly available critical care database
  - 61,532 ICU admissions from 46,520 patients
  - Data from 2001-2012 at Beth Israel Deaconess Medical Center
  - Fully de-identified (HIPAA-compliant)
  - Free access with registration

STEP 1: Create PhysioNet Account
  → Go to: https://physionet.org/
  → Click "Sign up"
  → Verify email
  → Complete profile

STEP 2: Request MIMIC-III Access
  → Go to: https://physionet.org/content/mimiciii/1.4/
  → Click "Request Access"
  → Read and sign data use agreement
  → Access approved within 24 hours

STEP 3: Install Requirements
  $ pip install wfdb pandas numpy

STEP 4: Authenticate
  $ python -c "import wfdb; wfdb.dl_database('mimic3wdb/p00/p001000')"
  → Enter PhysioNet username
  → Enter PhysioNet password
  → Password is stored securely in ~/.config/wfdb/credentials

STEP 5: Download NOTES Data
  Run: python src/download_mimic_data.py --download-notes

STEP 6: Prepare for Analysis
  Run: python src/download_mimic_data.py --prepare-data

═══════════════════════════════════════════════════════════════════════════════

WHY MIMIC-III?
  ✓ Already de-identified (HIPAA-compliant)
  ✓ Public domain (free access)
  ✓ Large sample size (60K+ admissions)
  ✓ Real clinical notes included
  ✓ Research-grade quality
  ✓ Extensively validated
  ✓ Published in Nature (credible source)

DATA INCLUDED:
  ✓ Admission notes
  ✓ Discharge summaries
  ✓ Clinical progress notes
  ✓ Procedures
  ✓ Diagnoses
  ✓ Medications
  ✓ Lab results
  ✓ Vital signs

PRIVACY ASSURANCE:
  ✓ All dates shifted by random offset (0-67 years)
  ✓ All patient/provider names removed
  ✓ All identifiers replaced with random numbers
  ✓ HIPAA Safe Harbor compliant
  ✓ IRB approved for research use

═══════════════════════════════════════════════════════════════════════════════

REFERENCE:
  Johnson AEW, Pollard TJ, Shen L, et al. (2016)
  "MIMIC-III, a freely accessible critical care database"
  Scientific Data 3:160035
  https://doi.org/10.1038/sdata.2016.35

═══════════════════════════════════════════════════════════════════════════════
        """
        print(instructions)
    
    def create_alternative_synthetic_enhanced(self, n_records):
        """
        If user can't access MIMIC-III, create enhanced synthetic data
        that mimics real clinical notes structure.
        """
        print("\n" + "="*70)
        print("Creating Enhanced Synthetic Medical Data")
        print("(Structured like real MIMIC-III notes)")
        print("="*70 + "\n")
        
        import random
        from datetime import datetime, timedelta
        
        # More realistic medical terminology from MIMIC-III
        chief_complaints = [
            "shortness of breath", "chest pain", "abdominal pain",
            "fever", "altered mental status", "sepsis", "pneumonia",
            "acute kidney injury", "hypotension", "tachycardia",
            "respiratory failure", "cardiac arrhythmia", "stroke",
            "myocardial infarction", "acute liver failure", "cough",
            "headache", "nausea", "fatigue", "dizziness"
        ]
        
        medical_history = [
            "hypertension", "diabetes mellitus", "congestive heart failure",
            "chronic obstructive pulmonary disease", "atrial fibrillation",
            "coronary artery disease", "chronic kidney disease",
            "asthma", "gastroesophageal reflux disease", "depression"
        ]
        
        current_medications = [
            "lisinopril", "metformin", "atorvastatin", "metoprolol",
            "warfarin", "levothyroxine", "omeprazole", "albuterol",
            "furosemide", "amoxicillin", "aspirin"
        ]
        
        vital_signs = [
            "BP 120/80 mmHg", "HR 85 bpm", "RR 16/min", "Temp 98.6°F",
            "BP 145/92 mmHg", "HR 102 bpm", "RR 22/min", "Temp 101.2°F",
            "BP 110/70 mmHg", "HR 72 bpm", "RR 14/min", "Temp 98.2°F"
        ]
        
        lab_results = [
            "WBC 7.2", "Hgb 13.4", "Plt 245", "BUN 18", "Cr 1.0",
            "Na 138", "K 4.2", "Cl 102", "CO2 24", "Glucose 125"
        ]
        
        assessments = [
            "acute infection", "sepsis syndrome", "acute respiratory distress",
            "acute coronary syndrome", "heart failure exacerbation",
            "metabolic acidosis", "acute kidney injury", "pneumonia"
        ]
        
        plans = [
            "Start broad-spectrum antibiotics",
            "Initiate mechanical ventilation",
            "Aggressive fluid resuscitation",
            "ICU admission for monitoring",
            "Consult cardiology",
            "Continuous cardiac monitoring",
            "Daily labs and imaging"
        ]
        
        records = []
        
        for i in range(n_records):
            age = random.randint(40, 85)
            gender = random.choice(["M", "F"])
            
            note = f"""
ADMISSION NOTE - {datetime.now().strftime('%Y-%m-%d')}

PATIENT ID: MIMIC_{i:06d}
AGE: {age} years
GENDER: {gender}

CHIEF COMPLAINT:
{random.choice(chief_complaints)}

HPI:
Patient is a {age}-year-old {gender} with history of {', '.join(random.sample(medical_history, 2))}.
Presents with {random.choice(chief_complaints)} for {random.randint(1,7)} days.

PMH: {', '.join(random.sample(medical_history, 3))}

Medications: {', '.join(random.sample(current_medications, 3))}

Vitals: {', '.join(random.sample(vital_signs, 4))}

Labs: {', '.join(random.sample(lab_results, 5))}

Assessment: {random.choice(assessments)}

Plan: {' '.join(random.sample(plans, 2))}
            """.strip()
            
            record = {
                "id": f"MIMIC_{i:06d}",
                "note": note,
                "age": age,
                "gender": gender,
                "admission_type": random.choice(["EMERGENCY", "URGENT", "ELECTIVE"])
            }
            records.append(record)
        
        print(f"✓ Created {len(records)} enhanced synthetic clinical notes")
        
        return records  # ✅ CHANGED: Return records list, not file path
    
    def prepare_mimic_notes(self, notes_csv_path):
        """
        Prepare downloaded MIMIC-III notes for analysis.
        Assumes notes.csv downloaded from PhysioNet.
        """
        try:
            import pandas as pd
        except ImportError:
            print("pandas required: pip install pandas")
            return None
        
        print(f"Loading MIMIC-III notes from: {notes_csv_path}")
        
        # Read notes
        df = pd.read_csv(notes_csv_path, nrows=5000)  # Limit to 5000 for memory
        
        # Filter for relevant note types
        relevant_types = ['Nursing', 'Physician', 'Discharge summary']
        df = df[df['CATEGORY'].isin(relevant_types)]
        
        # Create records
        records = []
        for idx, row in df.iterrows():
            if pd.isna(row['TEXT']):
                continue
            
            record = {
                "id": f"MIMIC_{int(row['HADM_ID'])}_{idx}",
                "note": str(row['TEXT'])[:2000],  # Limit note length
                "source": "MIMIC-III-real",
                "hadm_id": int(row['HADM_ID']),
                "note_type": row['CATEGORY'],
                "charttime": str(row['CHARTTIME'])
            }
            records.append(record)
        
        # Save
        output_file = os.path.join(self.output_dir, "mimic_real_notes.jsonl")
        with open(output_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        
        print(f"✓ Prepared {len(records)} real MIMIC-III notes")
        print(f"✓ Saved to: {output_file}")
        
        return output_file
    
    def generate_membership_inference_data(self, real_notes_path, synthetic_notes_path):
        """
        Generate train/test data for membership inference.
        Uses real and synthetic MIMIC-III notes.
        """
        from sklearn.model_selection import train_test_split
        
        # Load real notes
        with open(real_notes_path) as f:
            real_notes = [json.loads(line) for line in f]
        
        # Load synthetic notes
        with open(synthetic_notes_path) as f:
            synthetic_notes = [json.loads(line) for line in f]
        
        # Combine and create labels
        all_notes = real_notes + synthetic_notes
        labels = [0] * len(real_notes) + [1] * len(synthetic_notes)  # 0=real, 1=synthetic
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            all_notes, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Save
        def save_split(split_name, X, y):
            output_file = os.path.join(self.output_dir, f"mimic_membership_{split_name}.jsonl")
            with open(output_file, 'w') as f:
                for x, label in zip(X, y):
                    record = {
                        "id": x["id"],
                        "note": x["note"],
                        "source": x["source"],
                        "hadm_id": x["hadm_id"],
                        "label": label
                    }
                    f.write(json.dumps(record) + "\n")
            print(f"✓ Saved {len(X)} {split_name} data to: {output_file}")
        
        save_split("train", X_train, y_train)
        save_split("test", X_test, y_test)
    
    def split_and_write(self, records):
        """Split into train/test and save as JSONL."""
        import random
        
        random.seed(42)
        random.shuffle(records)
        
        n = len(records)
        split = int(0.7 * n)
        train = records[:split]
        test = records[split:]
        
        # Save training data
        train_file = os.path.join(self.output_dir, "mimic_synthetic_train.jsonl")
        with open(train_file, "w") as f:
            for r in train:
                f.write(json.dumps(r) + "\n")
        
        # Save test data
        test_file = os.path.join(self.output_dir, "mimic_synthetic_test.jsonl")
        with open(test_file, "w") as f:
            for r in test:
                f.write(json.dumps(r) + "\n")
        
        print(f"✓ Generated {len(train)} train and {len(test)} test MIMIC-like records")
        print(f"  Train: {train_file}")
        print(f"  Test:  {test_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare MIMIC-III data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show setup instructions
  python src/download_mimic_data.py --info
  
  # Create enhanced synthetic data (no PhysioNet access needed)
  python src/download_mimic_data.py --synthetic
  
  # Prepare downloaded MIMIC notes (after PhysioNet access)
  python src/download_mimic_data.py --prepare-data \\
    --notes-csv data/mimic_notes.csv
    
  # Generate membership inference train/test data
  python src/download_mimic_data.py --membership-inference \\
    --real-notes data/mimic_real_notes.jsonl \\
    --synthetic-notes data/mimic_synthetic_notes.jsonl
        """
    )
    
    parser.add_argument("--info", action="store_true", 
                       help="Show MIMIC-III setup instructions")
    parser.add_argument("--synthetic", action="store_true",
                       help="Create enhanced synthetic data (MIMIC-like)")
    parser.add_argument("--prepare-data", action="store_true",
                       help="Prepare downloaded MIMIC notes")
    parser.add_argument("--membership-inference", action="store_true",
                       help="Generate membership inference train/test data")
    parser.add_argument("--notes-csv", type=str,
                       help="Path to MIMIC-III notes.csv file")
    parser.add_argument("--real-notes", type=str,
                       help="Path to real MIMIC-III notes JSONL file")
    parser.add_argument("--synthetic-notes", type=str,
                       help="Path to synthetic MIMIC-III notes JSONL file")
    parser.add_argument("--output_dir", default="data/mimic",
                       help="Output directory")
    parser.add_argument("--n_records", type=int, default=1000,
                       help="Number of synthetic records to generate")
    
    args = parser.parse_args()
    
    prep = MIMICDataPreparation(args.output_dir)
    
    if args.info:
        prep.print_setup_instructions()
    
    elif args.synthetic:
        records = prep.create_alternative_synthetic_enhanced(args.n_records)  # ✅ Get records
        prep.split_and_write(records)  # ✅ Pass records to split_and_write
    
    elif args.prepare_data:
        if not args.notes_csv:
            print("Error: --notes-csv required with --prepare-data")
            sys.exit(1)
        prep.prepare_mimic_notes(args.notes_csv)
    
    elif args.membership_inference:
        if not args.real_notes or not args.synthetic_notes:
            print("Error: --real-notes and --synthetic-notes required with --membership-inference")
            sys.exit(1)
        prep.generate_membership_inference_data(args.real_notes, args.synthetic_notes)
    
    else:
        prep.print_setup_instructions()

if __name__ == "__main__":
    main()
