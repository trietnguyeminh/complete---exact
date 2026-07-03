import json, re

INPUT_NB = r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v4_fixed.ipynb"
OUTPUT_NB = r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v4_update.ipynb"

with open(INPUT_NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = cell['source']
    
    # 1. Update Config Cell
    if "CAU HINH -- Chinh sua o day" in source:
        # Replace paths
        source = re.sub(r'DATASET_PATH\s*=\s*".*?"', 'DATASET_PATH   = "C:/Users/Minh Triet/Downloads/Logic_Based_Educational_Queries (2).json"', source)
        source = re.sub(r'PHYSICS_DATASET_PATH\s*=\s*".*?"', 'PHYSICS_DATASET_PATH = "C:/Users/Minh Triet/Downloads/Physics_Problems.csv"', source)
        
        # Add Split Config before N_SAMPLES
        split_config = """
# --- Data Split Config ---
SEED = 42
TRAIN_RATIO = 0.85
VAL_RATIO   = 0.10
TEST_RATIO  = 0.05
RUN_ON_SPLIT = "test"  # 'test', 'train', 'val', hoac 'all'

"""
        source = source.replace("N_SAMPLES      = 411", split_config + "N_SAMPLES      = 411")
        cell['source'] = source
        
    # 2. Update Prompts and load_dataset (Stage 2)
    elif "STAGE 2 -- Ontology & Prompts" in source:
        # Update ANSWER_SYSTEM
        old_ans_sys = (
            "Ban la chuyen gia suy luan logic. Dua vao cac tien de FOL da duoc Z3 xac minh, hay tra loi cau hoi.\\n\"\n"
            "    'Tra loi JSON THUAN TUY: {\"answer\": \"A|B|C|D|Yes|No|Unknown\", \"reasoning\": \"...\"}\\n'\n"
        )
        new_ans_sys = (
            "Ban la chuyen gia suy luan logic. Dua vao cac tien de FOL da duoc Z3 xac minh, hay tra loi cau hoi.\\n\"\n"
            "    \"QUAN TRONG: Ban PHAI dua ra suy luan (reasoning) tung buoc TRUOC KHI dua ra dap an (answer).\\n\"\n"
            "    'Tra loi JSON THUAN TUY: {\"reasoning\": \"...\", \"answer\": \"A|B|C|D|Yes|No|Unknown\"}\\n'\n"
        )
        if '{"answer": "A|B|C|D|Yes|No|Unknown", "reasoning": "..."}' in source:
            source = source.replace(
                '\'Tra loi JSON THUAN TUY: {"answer": "A|B|C|D|Yes|No|Unknown", "reasoning": "..."}\\n\'',
                '\"QUAN TRONG: Ban PHAI dua ra suy luan (reasoning) tung buoc TRUOC KHI dua ra dap an (answer).\\n\"\n    \'Tra loi JSON THUAN TUY: {"reasoning": "...", "answer": "A|B|C|D|Yes|No|Unknown"}\\n\''
            )

        # Update PHYSICS_MC_SYSTEM
        if '{"reasoning": "...", "answer": "<answer>"}' in source and 'IMPORTANT: Write your step-by-step reasoning' not in source:
             source = source.replace(
                 '\'Output PURE JSON: {"reasoning": "...", "answer": "<answer>"}\\n\'',
                 '\"IMPORTANT: Write your step-by-step reasoning BEFORE giving the final answer.\\n\"\n    \'Output PURE JSON: {"reasoning": "...", "answer": "<answer>"}\\n\''
             )

        # Update load_dataset
        old_load = """def load_dataset(path, is_physics=False, max_samples=N_SAMPLES):
    if not path or not os.path.exists(path): return []
    if path.endswith(".csv"):
        with open(path, encoding="utf-8") as f: raw = list(csv.DictReader(f))
    else:
        with open(path, encoding="utf-8") as f: raw = json.load(f)
    out = raw[:max_samples]
    if is_physics and out:
        for s in out:
            s["premises-NL"] = []
            s["questions"]   = [s.get("question", "")]
            s["answers"]     = [str(s.get("answer", "Unknown"))]
            s["_unit"]       = s.get("unit", "")
            s["_is_physics"] = True
    return out"""

        new_load = """def load_dataset(path, is_physics=False, max_samples=N_SAMPLES, split_mode=RUN_ON_SPLIT):
    if not path or not os.path.exists(path): return []
    if path.endswith(".csv"):
        with open(path, encoding="utf-8") as f: raw = list(csv.DictReader(f))
    else:
        with open(path, encoding="utf-8") as f: raw = json.load(f)
    
    out = raw[:max_samples]
    
    # --- SPLIT LOGIC ---
    import random
    rng = random.Random(SEED)
    rng.shuffle(out)
    n = len(out)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    
    if split_mode == "train":
        out = out[:n_train]
    elif split_mode == "val":
        out = out[n_train:n_train+n_val]
    elif split_mode == "test":
        out = out[n_train+n_val:]
    # if split_mode == "all", keep out as is

    if is_physics and out:
        for s in out:
            s["premises-NL"] = []
            s["questions"]   = [s.get("question", "")]
            s["answers"]     = [str(s.get("answer", "Unknown"))]
            s["_unit"]       = s.get("unit", "")
            s["_is_physics"] = True
    return out"""
        source = source.replace(old_load, new_load)
        cell['source'] = source

with open(OUTPUT_NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Created notebook_v4_update.ipynb successfully!")
