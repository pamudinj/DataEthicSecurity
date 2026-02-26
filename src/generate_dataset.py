"""
Phase 2 — Synthetic Dataset Generation

Generates 17,000 synthetic clinical notes split into:
  - data/train.jsonl       (10,000 records: 95% common, 5% rare)
  - data/val.jsonl         (2,000 records: common only)
  - data/nonmember.jsonl   (5,000 records: ≥200 rare, rest common)

Each record: {"id": str, "text": str, "group": "common" | "rare"}
Only "text" is used for model training.
"group" and "id" are used for evaluation and split validation only.
"""

import json
import random
import argparse
import os

SEED = 42

# ---------------------------------------------------------------------------
# Vocabulary pools (condition-keyed)
# ---------------------------------------------------------------------------

COMMON_CONDITIONS = [
    "hypertension",
    "pneumonia",
    "type 2 diabetes",
    "lower back pain",
    "urinary tract infection",
    "gastroesophageal reflux disease",
    "asthma",
    "iron deficiency anemia",
    "migraine",
    "osteoarthritis",
]

RARE_CONDITIONS = [
    "Addison's disease",
    "Marfan syndrome",
    "autoimmune encephalitis",
    "primary biliary cholangitis",
    "systemic mastocytosis",
    "Castleman disease",
    "stiff person syndrome",
    "POEMS syndrome",
    "paraneoplastic cerebellar degeneration",
    "Erdheim-Chester disease",
]

SYMPTOMS = {
    "hypertension": ["headache", "dizziness", "blurred vision", "shortness of breath", "chest tightness"],
    "pneumonia": ["fever", "productive cough", "chest pain", "fatigue", "shortness of breath"],
    "type 2 diabetes": ["polyuria", "polydipsia", "blurred vision", "fatigue", "slow wound healing"],
    "lower back pain": ["lumbar stiffness", "radiating leg pain", "muscle spasm", "reduced range of motion", "tenderness"],
    "urinary tract infection": ["dysuria", "urinary frequency", "suprapubic pain", "cloudy urine", "low-grade fever"],
    "gastroesophageal reflux disease": ["heartburn", "regurgitation", "dysphagia", "chest discomfort", "chronic cough"],
    "asthma": ["wheezing", "shortness of breath", "chest tightness", "nocturnal cough", "exercise intolerance"],
    "iron deficiency anemia": ["fatigue", "pallor", "palpitations", "cold extremities", "exertional dyspnea"],
    "migraine": ["unilateral headache", "photophobia", "phonophobia", "nausea", "visual aura"],
    "osteoarthritis": ["joint pain", "morning stiffness", "crepitus", "swelling", "reduced mobility"],
    "Addison's disease": ["fatigue", "weight loss", "hyperpigmentation", "salt craving", "postural dizziness"],
    "Marfan syndrome": ["tall stature", "arachnodactyly", "pectus excavatum", "lens dislocation", "aortic dilation"],
    "autoimmune encephalitis": ["confusion", "seizures", "psychiatric symptoms", "memory impairment", "movement disorder"],
    "primary biliary cholangitis": ["pruritus", "fatigue", "right upper quadrant pain", "jaundice", "xanthomas"],
    "systemic mastocytosis": ["urticaria pigmentosa", "flushing", "abdominal cramping", "anaphylaxis", "bone pain"],
    "Castleman disease": ["lymphadenopathy", "fever", "night sweats", "fatigue", "weight loss"],
    "stiff person syndrome": ["progressive rigidity", "muscle spasms", "hyperlordosis", "anxiety", "hyperekplexia"],
    "POEMS syndrome": ["peripheral neuropathy", "organomegaly", "endocrinopathy", "monoclonal protein", "skin changes"],
    "paraneoplastic cerebellar degeneration": ["gait ataxia", "dysarthria", "nystagmus", "diplopia", "vertigo"],
    "Erdheim-Chester disease": ["bone pain", "diabetes insipidus", "exophthalmos", "retroperitoneal fibrosis", "xanthelasma"],
}

FINDINGS = {
    "hypertension": [
        "blood pressure 158/96 mmHg on two readings",
        "BP 162/100 mmHg with bilateral retinal arteriolar narrowing",
        "elevated BP 170/105 mmHg and mild left ventricular hypertrophy on ECG",
    ],
    "pneumonia": [
        "temperature 38.7°C, decreased breath sounds and dullness on percussion right base",
        "CXR shows right lower lobe consolidation, SpO2 94% on room air",
        "bilateral crackles on auscultation, CRP 87 mg/L",
    ],
    "type 2 diabetes": [
        "fasting glucose 9.4 mmol/L, HbA1c 8.2%",
        "random blood glucose 14.1 mmol/L, BMI 31",
        "HbA1c 9.0%, microalbuminuria on urinalysis",
    ],
    "lower back pain": [
        "MRI shows L4-L5 disc herniation with mild nerve root compression",
        "paraspinal muscle tenderness, pain 7/10 on VAS, reduced lumbar flexion",
        "straight leg raise positive at 40 degrees bilaterally",
    ],
    "urinary tract infection": [
        "urinalysis: positive nitrites, leukocyte esterase, >10 WBC/hpf",
        "urine culture pending, dipstick positive for nitrites and blood",
        "midstream urine cloudy, WBC 50/hpf, culture shows E. coli",
    ],
    "gastroesophageal reflux disease": [
        "endoscopy shows grade B esophagitis and hiatal hernia",
        "pH monitoring confirms abnormal acid exposure, DeMeester score 38",
        "esophageal manometry normal, 24h pH study positive",
    ],
    "asthma": [
        "peak flow 62% predicted, reversible obstruction post-bronchodilator",
        "FEV1/FVC 0.68, significant bronchodilator response of 18%",
        "bilateral expiratory wheeze, SpO2 96%, PEFR 55% predicted",
    ],
    "iron deficiency anemia": [
        "Hb 8.2 g/dL, MCV 71 fL, serum ferritin 6 ng/mL",
        "microcytic hypochromic anemia, ferritin <10, TIBC elevated",
        "Hb 9.0 g/dL, peripheral smear shows pencil cells and target cells",
    ],
    "migraine": [
        "neurological exam normal, no papilloedema, trigger diary confirms hormonal pattern",
        "MRI brain unremarkable, ICHD-3 criteria for migraine with aura met",
        "normal cranial nerve exam, photophobia confirmed, family history positive",
    ],
    "osteoarthritis": [
        "X-ray shows joint space narrowing and osteophytes at bilateral knees",
        "crepitus on passive ROM, Kellgren-Lawrence grade 3 on imaging",
        "MRI demonstrates cartilage loss and subchondral sclerosis at right hip",
    ],
    "Addison's disease": [
        "morning cortisol <83 nmol/L, ACTH stimulation test confirms adrenal insufficiency",
        "hyponatremia, hyperkalemia, low random cortisol, elevated ACTH",
        "anti-21-hydroxylase antibodies positive, cortisol 45 nmol/L post-synacthen",
    ],
    "Marfan syndrome": [
        "echocardiogram shows aortic root dilation at 42 mm, FBN1 mutation confirmed",
        "slit-lamp exam reveals bilateral ectopia lentis, arm span exceeds height",
        "aortic root Z-score +3.2, revised Ghent criteria met",
    ],
    "autoimmune encephalitis": [
        "CSF pleocytosis, anti-NMDAR antibodies positive in serum and CSF",
        "EEG shows delta brush pattern, MRI FLAIR hyperintensity in medial temporal lobes",
        "anti-LGI1 antibodies positive, MRI shows hippocampal T2 signal change",
    ],
    "primary biliary cholangitis": [
        "ALP 420 U/L, GGT elevated, anti-mitochondrial antibodies M2 positive",
        "liver biopsy shows florid duct lesion, AMA titre 1:640",
        "ALP 380 U/L, AMA positive, fibroscan shows F2 fibrosis",
    ],
    "systemic mastocytosis": [
        "bone marrow biopsy shows mast cell aggregates >15 cells, KIT D816V mutation detected",
        "serum tryptase 48 ng/mL, skin biopsy confirms urticaria pigmentosa",
        "KIT D816V positive, tryptase 62 ng/mL, DEXA shows osteoporosis",
    ],
    "Castleman disease": [
        "CT shows mediastinal lymphadenopathy, biopsy reveals hyaline-vascular variant",
        "IL-6 elevated, PET-CT shows multicentric FDG-avid nodes, HHV-8 negative",
        "lymph node biopsy plasma cell variant, VEGF elevated, POEMS excluded",
    ],
    "stiff person syndrome": [
        "anti-GAD antibodies titre >2000 IU/mL, EMG shows continuous motor unit activity",
        "spinal MRI normal, anti-GAD 65 strongly positive, clinically confirmed",
        "lumbar paraspinal EMG confirms continuous firing at rest, anti-amphiphysin negative",
    ],
    "POEMS syndrome": [
        "serum VEGF markedly elevated, monoclonal lambda light chain, nerve conduction confirms polyneuropathy",
        "bone survey shows sclerotic lesions, M-protein IgA lambda, hepatosplenomegaly on CT",
        "VEGF 1840 pg/mL, skin biopsy shows angiomatosis, endocrine work-up shows hypothyroidism",
    ],
    "paraneoplastic cerebellar degeneration": [
        "anti-Yo antibodies positive, CT chest reveals right lung mass",
        "MRI cerebellum shows atrophy, anti-Hu positive, SCLC confirmed on biopsy",
        "anti-CASPR2 antibodies detected, PET-CT identifies thymoma",
    ],
    "Erdheim-Chester disease": [
        "BRAF V600E mutation detected, PET-CT shows periaortic soft tissue infiltration",
        "bilateral symmetric sclerosis of long bones on X-ray, CD68+/CD1a- histiocytes on biopsy",
        "MRI shows retroorbital and retroperitoneal infiltration, BRAF wild-type, MEK pathway activation",
    ],
}

PLANS = {
    "hypertension": [
        "initiate amlodipine 5 mg daily, low-sodium diet, repeat BP in 4 weeks",
        "commence lisinopril 10 mg daily, lifestyle modification counselling, renal function check in 2 weeks",
        "dual therapy with perindopril and indapamide, 24h ambulatory BP monitoring arranged",
    ],
    "pneumonia": [
        "admit for IV amoxicillin-clavulanate, supplemental oxygen, repeat CXR in 48h",
        "outpatient amoxicillin 500 mg TDS for 5 days, safety netting, review if not improving",
        "azithromycin for atypical cover, oral rehydration, review in 48 hours",
    ],
    "type 2 diabetes": [
        "commence metformin 500 mg BD with titration, dietitian referral, HbA1c in 3 months",
        "intensify to SGLT2 inhibitor, refer to diabetes educator, ophthalmology screening",
        "add GLP-1 agonist for weight management, podiatry referral, optimise statin",
    ],
    "lower back pain": [
        "physiotherapy referral, naproxen 500 mg BD with food, reassess in 6 weeks",
        "short course diazepam for spasm, heat therapy, core strengthening exercises",
        "pain clinic referral, trial of gabapentin, MRI-guided steroid injection considered",
    ],
    "urinary tract infection": [
        "trimethoprim 200 mg BD for 7 days, encourage fluid intake, urine MC&S follow-up",
        "nitrofurantoin 100 mg BD for 5 days, avoid in renal impairment, review culture result",
        "fosfomycin single dose, prophylaxis considered if recurrent, repeat MSU in 1 week",
    ],
    "gastroesophageal reflux disease": [
        "omeprazole 20 mg daily before breakfast, avoid late meals and alcohol, elevate head of bed",
        "lansoprazole 30 mg daily, weight loss advice, step-down therapy planned after 8 weeks",
        "refer for laparoscopic Nissen fundoplication if refractory, continue PPI in interim",
    ],
    "asthma": [
        "salbutamol 100 mcg PRN, commence beclomethasone 200 mcg BD, written action plan provided",
        "step up to low-dose ICS/LABA, check inhaler technique, smoking cessation advice",
        "prednisolone 30 mg for 5 days, nebulised salbutamol, consider specialist referral",
    ],
    "iron deficiency anemia": [
        "ferrous sulfate 200 mg BD, dietary iron advice, recheck Hb in 4 weeks",
        "IV iron infusion (Ferinject), investigate for occult GI blood loss, colonoscopy arranged",
        "treat underlying cause, ferrous fumarate 210 mg TDS, repeat FBC in 6 weeks",
    ],
    "migraine": [
        "sumatriptan 50 mg at onset, avoid known triggers, headache diary maintained",
        "commence topiramate 25 mg nocte for prophylaxis, avoid opioids, neurology referral",
        "naproxen sodium 550 mg for acute attacks, trial amitriptyline prophylaxis, lifestyle review",
    ],
    "osteoarthritis": [
        "paracetamol 1 g QDS regular, physiotherapy, weight loss programme referral",
        "topical diclofenac gel, hydrotherapy, consider intra-articular steroid injection",
        "refer for orthopaedic review for total knee replacement, oral NSAIDs short-term",
    ],
    "Addison's disease": [
        "hydrocortisone 10/5/5 mg replacement, fludrocortisone 100 mcg daily, sick day rules education",
        "register with medic alert, IM hydrocortisone kit prescribed, endocrinology follow-up",
        "stress dosing protocol explained, annual bone density and electrolyte review",
    ],
    "Marfan syndrome": [
        "losartan 50 mg daily, cardiothoracic surgery referral, avoid contact sports",
        "annual echocardiogram surveillance, beta-blocker prophylaxis, genetic counselling arranged",
        "elective aortic root replacement recommended, ophthalmology review, scoliosis monitoring",
    ],
    "autoimmune encephalitis": [
        "IV methylprednisolone 1 g for 5 days, IVIG 2 g/kg, oncology screen for underlying malignancy",
        "rituximab initiated, EEG monitoring, tumour surveillance CT arranged",
        "plasma exchange, transfer to neuro-ITU, long-term immunosuppression with mycophenolate",
    ],
    "primary biliary cholangitis": [
        "ursodeoxycholic acid 13–15 mg/kg/day, annual LFT and liver stiffness monitoring",
        "add obeticholic acid for inadequate UDCA response, cholestyramine for pruritus",
        "hepatology referral, fibroscan every 2 years, assess for liver transplant candidacy",
    ],
    "systemic mastocytosis": [
        "antihistamines H1 and H2, epinephrine auto-injector prescribed, avoid triggers",
        "midostaurin commenced, bisphosphonate for osteoporosis, haematology co-management",
        "mast cell stabiliser sodium cromoglycate, regular tryptase monitoring, bone marrow review in 2 years",
    ],
    "Castleman disease": [
        "siltuximab every 3 weeks, PET-CT response assessment at 6 cycles",
        "rituximab plus cyclophosphamide, dexamethasone, etoposide protocol initiated",
        "surgical resection for unicentric disease, surveillance CT every 6 months",
    ],
    "stiff person syndrome": [
        "diazepam 5 mg TDS, baclofen 10 mg TDS, IVIG 2 g/kg monthly",
        "initiate rituximab for refractory disease, physiotherapy for gait rehabilitation",
        "intrathecal baclofen pump referral, pregabalin for pain, psychological support",
    ],
    "POEMS syndrome": [
        "bortezomib-based chemotherapy, autologous stem cell transplant assessment, VEGF monitoring",
        "lenalidomide and dexamethasone, radiation for solitary plasmacytoma, cardiology review",
        "thalidomide and dexamethasone, supportive care with diuretics, peripheral neuropathy management",
    ],
    "paraneoplastic cerebellar degeneration": [
        "treat underlying malignancy, IV methylprednisolone, IVIG for cerebellar symptoms",
        "plasma exchange, oncology co-management, physical rehabilitation for ataxia",
        "rituximab for anti-CASPR2 associated disease, PET-CT for occult tumour, speech therapy",
    ],
    "Erdheim-Chester disease": [
        "vemurafenib for BRAF V600E mutation, PET-CT response at 3 months",
        "cobimetinib and vemurafenib combination, multidisciplinary review, diabetes insipidus managed with desmopressin",
        "interferon-alpha for BRAF wild-type, annual surveillance imaging, ophthalmology monitoring",
    ],
}

# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------

def make_record(record_id: int, condition: str, group: str, rng: random.Random) -> dict:
    """Build one clinical note record."""
    age = rng.randint(18, 85)
    gender = rng.choice(["male", "female", "non-binary"])
    symptoms = ", ".join(rng.sample(SYMPTOMS[condition], k=2))
    findings = rng.choice(FINDINGS[condition])
    plan = rng.choice(PLANS[condition])

    text = (
        f"Patient {age}-year-old {gender} presents with {symptoms}. "
        f"Findings: {findings}. "
        f"Diagnosis: {condition}. "
        f"Plan: {plan}."
    )

    return {
        "id": f"patient_{record_id:05d}",
        "text": text,
        "group": group,
    }


def generate_unique_records(n: int, group: str, conditions: list, start_id: int,
                            rng: random.Random, existing_texts: set) -> list:
    """
    Generate `n` unique records for the given group.
    Re-samples until the text is not in `existing_texts`, then adds it.
    This guarantees strict disjointness across splits.
    """
    records = []
    idx = start_id
    attempts = 0
    max_attempts = n * 100  # safety cap

    while len(records) < n:
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {n} unique records after {max_attempts} attempts. "
                "Consider expanding vocabulary or reducing dataset size."
            )
        condition = rng.choice(conditions)
        record = make_record(idx, condition, group, rng)
        if record["text"] not in existing_texts:
            existing_texts.add(record["text"])
            records.append(record)
            idx += 1
        attempts += 1

    return records, idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2 — Synthetic Dataset Generation")
    parser.add_argument("--out_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--n_train", type=int, default=10_000)
    parser.add_argument("--n_val", type=int, default=2_000)
    parser.add_argument("--n_nonmember", type=int, default=5_000)
    parser.add_argument("--rare_train_frac", type=float, default=0.05,
                        help="Fraction of training set that is rare")
    parser.add_argument("--rare_nonmember_min", type=int, default=200,
                        help="Minimum rare records in non-member pool")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Global set of all generated texts — enforces disjointness at generation time
    all_texts: set = set()

    # --- Train split (95% common, 5% rare) ---
    n_train_rare = int(args.n_train * args.rare_train_frac)    # 500
    n_train_common = args.n_train - n_train_rare               # 9500
    print(f"Generating train split ({n_train_common} common + {n_train_rare} rare)...")
    train_common, next_id = generate_unique_records(n_train_common, "common", COMMON_CONDITIONS, 0, rng, all_texts)
    train_rare, next_id = generate_unique_records(n_train_rare, "rare", RARE_CONDITIONS, next_id, rng, all_texts)
    train_records = train_common + train_rare
    rng.shuffle(train_records)

    # --- Val split (common only) ---
    print(f"Generating val split ({args.n_val} common)...")
    val_records, next_id = generate_unique_records(args.n_val, "common", COMMON_CONDITIONS, next_id, rng, all_texts)

    # --- Non-member split (≥200 rare, rest common) ---
    n_nm_rare = max(args.rare_nonmember_min, int(args.n_nonmember * 0.05))
    n_nm_common = args.n_nonmember - n_nm_rare
    print(f"Generating non-member split ({n_nm_common} common + {n_nm_rare} rare)...")
    nm_common, next_id = generate_unique_records(n_nm_common, "common", COMMON_CONDITIONS, next_id, rng, all_texts)
    nm_rare, next_id = generate_unique_records(n_nm_rare, "rare", RARE_CONDITIONS, next_id, rng, all_texts)
    nonmember_records = nm_common + nm_rare
    rng.shuffle(nonmember_records)

    # --- Disjoint assertion (guaranteed by construction, kept as safety net) ---
    train_texts = {r["text"] for r in train_records}
    val_texts = {r["text"] for r in val_records}
    nm_texts = {r["text"] for r in nonmember_records}

    assert train_texts.isdisjoint(val_texts), "OVERLAP: train ∩ val"
    assert train_texts.isdisjoint(nm_texts), "OVERLAP: train ∩ nonmember"
    assert val_texts.isdisjoint(nm_texts), "OVERLAP: val ∩ nonmember"

    # --- Write JSONL files ---
    splits = {
        "train.jsonl": train_records,
        "val.jsonl": val_records,
        "nonmember.jsonl": nonmember_records,
    }

    for filename, records in splits.items():
        path = os.path.join(args.out_dir, filename)
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"✓ Wrote {len(records):,} records → {path}")

    # --- Summary ---
    train_rare_count = sum(1 for r in train_records if r["group"] == "rare")
    nm_rare_count = sum(1 for r in nonmember_records if r["group"] == "rare")

    print("\n--- Dataset Summary ---")
    print(f"  Train:      {len(train_records):,}  ({train_rare_count} rare / {len(train_records)-train_rare_count} common)")
    print(f"  Val:        {len(val_records):,}  (common only)")
    print(f"  Non-member: {len(nonmember_records):,}  ({nm_rare_count} rare / {len(nonmember_records)-nm_rare_count} common)")
    print(f"  Total:      {len(train_records)+len(val_records)+len(nonmember_records):,}")
    print(f"  Seed:       {args.seed}")
    print(f"  Disjoint splits: ✓ verified")


if __name__ == "__main__":
    main()
