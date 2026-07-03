#!/usr/bin/env python3
"""
Build script: Create notebook_v2_physics.ipynb from original notebook.
Applies physics pipeline fix without modifying logic pipeline.
"""
import json, sys, copy
sys.stdout.reconfigure(encoding='utf-8')

# ================================================================
# LOAD ORIGINAL
# ================================================================
ORIG = r'C:\Users\Minh Triet\Desktop\SKIBIDI\notebooka4ea9824c7.ipynb'
OUT  = r'C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v2_physics.ipynb'

with open(ORIG, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Original: {len(cells)} cells")

# ================================================================
# Helper: create a clean code cell (no outputs)
# ================================================================
def make_cell(source):
    return {
        "cell_type": "code",
        "source": source,
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None
    }

# ================================================================
# CELL 0: Docstring -- update description
# ================================================================
cells[0] = make_cell(
    cells[0]['source'].replace(
        "vllm_pipeline.py -- Neuro-Symbolic Pipeline with vLLM (Batch Inference)",
        "vllm_pipeline_v2.py -- Neuro-Symbolic Pipeline with vLLM + Physics Direct Solver"
    )
)

# ================================================================
# CELL 1: Kaggle T4 fix -- keep as-is (clear outputs only)
# ================================================================
cells[1]['outputs'] = []
cells[1]['execution_count'] = None

# ================================================================
# CELL 2: Stage 0 Install -- keep as-is (clear outputs only)
# ================================================================
cells[2]['outputs'] = []
cells[2]['execution_count'] = None

# ================================================================
# CELL 3: CONFIG -- update GPU_MEMORY_UTIL, MAX_NEW_TOKENS, add physics config
# ================================================================
cells[3] = make_cell(r"""
# ==================================================================
# CAU HINH -- Chinh sua o day
# ==================================================================
QWEN_MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH   = "/kaggle/input/datasets/nguyenminhtric/test-pipeline/Logic_Based_Educational_Queries (2).json"
PHYSICS_DATASET_PATH = "/kaggle/input/datasets/nguyenminhtric/test-pipeline/Physics_Problems.csv"
N_SAMPLES      = 411
MAX_RETRIES    = 3
OUTPUT_PATH    = "/kaggle/working/pipeline_results_vllm.json"
MAX_NEW_TOKENS = 4096      # token toi da cho formalization
ANS_MAX_TOKENS = 600       # token toi da cho answer extraction

# vLLM Config
TENSOR_PARALLEL  = min(N_GPUS, 2)  # Tu dong dung so GPU co san
MAX_MODEL_LEN    = 8192            # Giam tu 32k de tiet kiem KV cache
GPU_MEMORY_UTIL  = 0.85            # % VRAM cho vLLM
DTYPE            = "half"          # fp16 cho T4 (khong ho tro bf16)

# ===== PHYSICS PIPELINE CONFIG =====
PHYSICS_MODE     = "direct"        # "direct" = skip FOL/Z3, dung CoT solver
                                   # (Logic dataset luon dung FOL pipeline)
PHYSICS_MAX_TOKENS = 1024          # token cho physics solver (can nhieu buoc tinh)
PHYSICS_TOLERANCE  = 0.05          # 5% relative tolerance cho numeric matching
# ==================================================================

print(f"Config OK")
print(f"  Model         : {QWEN_MODEL_ID}")
print(f"  Tensor Parallel: {TENSOR_PARALLEL} GPU(s)")
print(f"  Max Model Len : {MAX_MODEL_LEN}")
print(f"  GPU Mem Util  : {GPU_MEMORY_UTIL}")
print(f"  N_SAMPLES     : {N_SAMPLES}  MAX_RETRIES: {MAX_RETRIES}")
print(f"  Physics Mode  : {PHYSICS_MODE}")

""")

# ================================================================
# CELL 4: Stage 1 Load vLLM -- keep as-is (clear outputs only)
# ================================================================
cells[4]['outputs'] = []
cells[4]['execution_count'] = None

# ================================================================
# CELL 5: Stage 2 -- Ontology, Prompts, Dataset Loader, Helpers
# Add physics prompts + fix load_dataset
# ================================================================
cells[5] = make_cell(r'''
# ==================================================================
# STAGE 2 -- Dual-Layer Ontology & Data Loading
# ==================================================================

GLOBAL_ONTOLOGY = {
    "quantifiers": ["forall", "exists"],
    "logical_operators": ["and", "or", "implies", "iff", "not"],
    "ast_node_types": ["quantifier", "connective", "predicate", "variable", "constant"],
}

GLOBAL_ONTOLOGY_TEXT = """
## GLOBAL ONTOLOGY -- BAT BUOC TUAN THU (KHONG duoc sua doi)

### Luong tu (Quantifiers):
  forall  -> forall  (voi moi)
  exists  -> exists  (ton tai)

### Toan tu logic (Logical Operators):
  and     -> AND
  or      -> OR
  implies -> IMPLIES (keo theo)
  iff     -> IFF (tuong duong)
  not     -> NOT (phu dinh)

### So do 4 loai AST Node (phai dung DUNG nhu duoi):
  quantifier : { "type":"quantifier",  "operator":"forall|exists",
                 "bound_variables":["x",...], "body":{...} }
  connective : { "type":"connective",  "operator":"and|or|implies|iff|not",
                 "operands":[{...},{...},...] }
  predicate  : { "type":"predicate",   "name":"PredicateName",
                 "arguments":["x","y",...] }
  variable   : { "type":"variable",    "name":"x" }
  constant   : { "type":"constant",    "name":"SomeName" }

### QUY TAC CUNG (vi pham -> Z3 loi):
  1. Chi dung 4 node type tren, KHONG sang tao them
  2. 'not' chi co DUNG 1 operand
  3. 'implies' co DUNG 2 operands (ve trai, ve phai)
  4. bound_variables phai la list (du chi 1 bien)
  5. Bien dung: x, y, z (lowercase); hang so: PascalCase
"""

# -- Logic Prompt Templates (KHONG THAY DOI) --

FORMALIZATION_SYSTEM = (
    "Ban la chuyen gia logic hinh thuc (FOL). Nhiem vu:\n"
    "  Buoc 1: Tao LOCAL ONTOLOGY -- danh sach cac Predicate quan trong\n"
    "  Buoc 2: Chuyen TUNG tien de thanh cay AST JSON de quy\n\n"
    + GLOBAL_ONTOLOGY_TEXT
    + "\n"
    "Output JSON THUAN TUY (khong markdown, khong text thua):\n"
    '{\n'
    '  "step1_local_ontology": [\n'
    '    {"predicate": "TenPredicate", "arity": 1, "description": "Mo ta ngan"}\n'
    '  ],\n'
    '  "step2_premises_ast": [\n'
    '    {"premise_id": 0, "source_nl": "...", "ast": { <AST node> }}\n'
    '  ]\n'
    '}\n'
)

CORRECTION_SYSTEM = (
    "Ban la chuyen gia sua loi FOL. He thong Z3 da phat hien loi.\n"
    "Nhiem vu: sua lai TOAN BO (Buoc 1 + Buoc 2) de khong con loi compile.\n\n"
    + GLOBAL_ONTOLOGY_TEXT
    + "\n"
    "Loi hay gap:\n"
    "  - Arity khong nhat quan\n"
    "  - Variable chua khai bao trong bound_variables\n"
    "  - Predicate khong co trong Local Ontology (hallucination)\n"
    "  - 'not' co nhieu hon 1 operand\n"
    "  - 'implies' khong du 2 operands\n\n"
    "Output JSON thuan tuy -- format GIONG HET lan dau.\n"
)

ANSWER_SYSTEM = (
    "Ban la chuyen gia suy luan logic.\n"
    "Dua vao cac tien de FOL da duoc xac minh boi Z3, hay tra loi cau hoi.\n\n"
    "Quy tac:\n"
    "  - Trac nghiem (A/B/C/D): tra ve 1 chu cai HOA\n"
    '  - Yes/No: tra ve "Yes", "No", hoac "Unknown"\n'
    "  - Suy luan chat che, khong suy doan ngoai pham vi\n\n"
    "Output JSON THUAN TUY:\n"
    '{"answer": "A|B|C|D|Yes|No|Unknown", "reasoning": "giai thich ngan"}\n'
)

# ===== PHYSICS-SPECIFIC PROMPTS (MOI) =====

PHYSICS_SOLVER_SYSTEM = (
    "You are an expert physics problem solver. "
    "Solve the given problem step-by-step with clear calculations.\n\n"
    "Rules:\n"
    "  - Identify all given quantities with correct units\n"
    "  - State the relevant physics formula(s)\n"
    "  - Show unit conversions explicitly (e.g. cm -> m, uF -> F)\n"
    "  - Compute step by step, show intermediate results\n"
    "  - Give the FINAL numerical answer (number only, no unit in answer field)\n\n"
    "Output PURE JSON (no markdown, no extra text):\n"
    '{"steps": ["step1...", "step2...", ...], '
    '"answer": "<number or math expression>", '
    '"unit": "<unit>"}\n'
)

PHYSICS_MC_SYSTEM = (
    "You are an expert physics problem solver. "
    "Solve the given multiple-choice problem.\n\n"
    "Rules:\n"
    "  - Work through the physics step by step\n"
    "  - Select the correct option\n"
    "  - If options are labeled A/B/C/D, return the letter\n"
    "  - If options are text, return the correct text\n\n"
    "Output PURE JSON (no markdown):\n"
    '{"reasoning": "step-by-step solution", "answer": "<answer>"}\n'
)

print("Prompt templates san sang (Logic + Physics)")


# -- Dataset Loader (UPDATED for physics) --

def load_dataset(path, is_physics=False, max_samples=N_SAMPLES):
    """Load va chuan hoa dataset, ho tro JSON va CSV."""
    if not path or not os.path.exists(path):
        print(f"[WARN] Dataset path not found: {path}")
        return []
    if path.endswith(".csv"):
        with open(path, encoding="utf-8") as f:
            raw = list(csv.DictReader(f))
    else:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    out = raw[:max_samples]

    if is_physics and out:
        for s in out:
            # ===== FIX: Proper field mapping for physics =====
            # Physics problems are NOT logic premises -- don't force FOL
            s["premises-NL"] = []                          # Empty: no FOL
            s["questions"]   = [s.get("question", "")]
            s["answers"]     = [str(s.get("answer", "Unknown"))]
            s["_unit"]       = s.get("unit", "")           # Preserve unit
            s["_cot"]        = s.get("cot", "")            # Preserve CoT
            s["_is_physics"] = True                        # Flag for routing

    return out


# -- JSON Parser --

def safe_json(text):
    """Trich xuat JSON tu response -- robust multi-strategy parser."""
    text = text.strip()
    # 1) Direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) Code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    # 3) Brace matching
    start = text.find("{")
    if start >= 0:
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return {}


# -- Batch Generate Helper --

def batch_generate(prompt_pairs, max_tokens=MAX_NEW_TOKENS):
    """
    Batch generate using vLLM.

    Args:
        prompt_pairs: list of (system_msg, user_msg) tuples
        max_tokens: max new tokens per response

    Returns:
        list of response strings
    """
    # Format prompts using chat template
    formatted = []
    for sys_msg, usr_msg in prompt_pairs:
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": usr_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)

    params = SamplingParams(
        temperature=0.05,
        max_tokens=max_tokens,
        repetition_penalty=1.1,
    )

    outputs = llm.generate(formatted, params)

    # Sort by request_id to preserve order
    outputs_sorted = sorted(outputs, key=lambda x: int(x.request_id))
    return [o.outputs[0].text.strip() for o in outputs_sorted]


print("batch_generate() san sang")

''')

# ================================================================
# CELL 6: Stage 3 -- Z3 Compiler -- keep EXACTLY as-is
# ================================================================
cells[6]['outputs'] = []
cells[6]['execution_count'] = None

# ================================================================
# CELL 7: Stage 4 -- Batch Pipeline + Physics Pipeline
# ================================================================
cells[7] = make_cell(r'''
# ==================================================================
# STAGE 4 -- Batch Pipeline Orchestrator + Physics Direct Solver
# ==================================================================

@dataclass
class PipelineResult:
    sample_id: int
    status: str = "pending"
    z3_status: str = "pending"
    z3_compiled: int = 0
    z3_total: int = 0
    z3_attempts: int = 0
    z3_errors: list = field(default_factory=list)
    hallucination_warn: list = field(default_factory=list)
    local_ontology: list = field(default_factory=list)
    premises_ast: list = field(default_factory=list)
    predicted_answers: list = field(default_factory=list)
    ground_truth: list = field(default_factory=list)
    total_questions: int = 0
    correct_count: int = 0
    time_sec: float = 0.0
    error_log: list = field(default_factory=list)


# ===================================================================
# LOGIC PIPELINE HELPERS (KHONG THAY DOI)
# ===================================================================

def _make_formalization_user(sample):
    """Tao user prompt cho formalization."""
    premises = sample["premises-NL"]
    numbered = "\n".join(f"Premise {i+1}: {p}" for i, p in enumerate(premises))
    return (
        "Hay hinh thuc hoa cac tien de sau theo dung quy trinh 2 buoc.\n\n"
        + numbered
        + "\n\nNho:\n"
        "  Buoc 1: khai bao Local Ontology\n"
        "  Buoc 2: sinh cay AST JSON de quy cho tung premise\n"
        "  Chi tra ve JSON thuan tuy."
    )


def _make_correction_user(sample, result):
    """Tao user prompt cho correction (khi Z3 bao loi)."""
    premises = sample["premises-NL"]
    numbered = "\n".join(f"Premise {i+1}: {p}" for i, p in enumerate(premises))
    z3_errors = "\n".join(result.z3_errors[:5]) or "(khong co loi cu the)"
    hall_errs = "\n".join(result.hallucination_warn) if result.hallucination_warn else "(khong co)"
    prev_local = json.dumps(result.local_ontology, ensure_ascii=False, indent=2)

    return (
        "He thong Z3 da phat hien loi khi compile cay AST.\n\n"
        "===== LOI TU Z3 =====\n"
        f"Z3 status: {result.z3_status}\n"
        f"Compiled: {result.z3_compiled} / {result.z3_total}\n\n"
        f"Loi chi tiet:\n{z3_errors}\n\n"
        f"Hallucination:\n{hall_errs}\n\n"
        "===== LOCAL ONTOLOGY CU =====\n"
        f"{prev_local}\n\n"
        "===== PREMISES GOC =====\n"
        f"{numbered}\n\n"
        "Sua lai TOAN BO (Buoc 1 + Buoc 2). Chi tra ve JSON thuan tuy."
    )


def _make_answer_user(sample, fol_context, question, q_idx):
    """Tao user prompt cho answer extraction."""
    premises = sample["premises-NL"]
    p_text = "\n".join(f"P{i+1}: {p}" for i, p in enumerate(premises))
    fol_text = "\n".join(f"FOL P{i+1}: {f}" for i, f in enumerate(fol_context))
    return (
        "## Tien de (Natural Language):\n"
        f"{p_text}\n\n"
        "## Tien de (FOL da xac minh qua Z3):\n"
        f"{fol_text}\n\n"
        f"## Cau hoi {q_idx + 1}:\n"
        f"{question}\n\n"
        "Tra loi JSON thuan tuy."
    )


def _process_formalization(result, raw_response):
    """Parse formalization response va chay Z3."""
    formalization = safe_json(raw_response)
    local_onto = formalization.get("step1_local_ontology", [])
    premises_ast = formalization.get("step2_premises_ast", [])

    result.local_ontology = local_onto
    result.premises_ast = premises_ast

    if not premises_ast:
        result.z3_status = "no_ast"
        result.error_log.append("step2_premises_ast rong")
        return

    hw = hallucination_check(local_onto, premises_ast)
    z3_info = verify_with_z3(premises_ast)

    result.hallucination_warn = hw
    result.z3_status = z3_info["status"]
    result.z3_errors = z3_info.get("errors", [])
    result.z3_compiled = z3_info.get("compiled_count", 0)
    result.z3_total = z3_info.get("total_count", 0)


# ===================================================================
# PHYSICS PIPELINE (MOI -- bypass FOL/Z3)
# ===================================================================

def _normalize_number(s):
    """Normalize numeric string for comparison.
    Handles: plain numbers, scientific notation (24.45 x 10^-3), etc.
    """
    s = str(s).strip()
    
    # Remove trailing unit text if accidentally included
    # e.g. "45 J" -> "45", "0.05 N" -> "0.05"
    s = re.sub(r'\s*[A-Za-z/]+\s*$', '', s).strip()
    
    # Handle "x 10^n" scientific notation variants
    # "24.45 × 10^-3", "24.45 x 10^(-3)", "24.45*10^-3"
    m = re.match(
        r'^([+-]?[\d.]+)\s*[×xX\*]\s*10\s*\^?\s*\(?\s*([+-]?\d+)\s*\)?\s*$',
        s
    )
    if m:
        try:
            mantissa = float(m.group(1))
            exp = int(m.group(2))
            return mantissa * (10 ** exp)
        except:
            pass
    
    # Handle plain number
    try:
        return float(s)
    except:
        pass
    
    return None


def _physics_answer_match(predicted, ground_truth, tolerance=None):
    """
    Flexible matching for physics answers.
    Handles: exact match, numeric tolerance, scientific notation, text.
    """
    if tolerance is None:
        tolerance = PHYSICS_TOLERANCE
    
    p = str(predicted).strip()
    g = str(ground_truth).strip()
    
    # 1) Exact string match (case-insensitive)
    if p.lower() == g.lower():
        return True
    
    # 2) Numeric comparison with relative tolerance
    p_num = _normalize_number(p)
    g_num = _normalize_number(g)
    
    if p_num is not None and g_num is not None:
        if g_num == 0:
            return abs(p_num) < 1e-9
        rel_error = abs(p_num - g_num) / abs(g_num)
        return rel_error <= tolerance
    
    # 3) Text containment (for text answers like "maximum", "decreases")
    p_low = p.lower()
    g_low = g.lower()
    if len(g_low) > 3 and (g_low in p_low or p_low in g_low):
        return True
    
    # 4) Strip units from prediction and retry numeric match
    p_clean = re.sub(r'[^\d.eE+\-×xX\*\^()]+', '', p)
    g_clean = re.sub(r'[^\d.eE+\-×xX\*\^()]+', '', g)
    if p_clean and g_clean:
        p2 = _normalize_number(p_clean)
        g2 = _normalize_number(g_clean)
        if p2 is not None and g2 is not None and g2 != 0:
            if abs(p2 - g2) / abs(g2) <= tolerance:
                return True
    
    return False


def _run_physics_pipeline(samples, results, N, t0_all, dataset_name):
    """
    Physics pipeline: Skip FOL/Z3, dung Qwen direct solve.
    Flow: Question -> Qwen CoT solve -> numerical answer -> flexible matching.
    """
    print(f"\n{'=' * 60}")
    print(f"  [Physics Mode] Direct Solve -- {N} samples")
    print(f"  (FOL/Z3 bypassed -- using numerical solver)")
    print(f"{'=' * 60}")

    # ===== BUILD PROMPTS =====
    solve_prompts = []
    for i, s in enumerate(samples):
        question = s["questions"][0]
        unit = s.get("_unit", "")
        
        # Check if multiple-choice
        is_mc = any(opt in question for opt in [
            "A.", "B.", "C.", "D.", "(A)", "(B)", "(C)", "(D)"
        ])
        
        if is_mc:
            sys_msg = PHYSICS_MC_SYSTEM
            user_msg = (
                f"Problem:\n{question}\n\n"
                "Solve step-by-step and select the correct answer."
            )
        else:
            sys_msg = PHYSICS_SOLVER_SYSTEM
            unit_hint = f"\nExpected answer unit: {unit}" if unit and unit != "—" else ""
            user_msg = (
                f"Problem:\n{question}"
                f"{unit_hint}\n\n"
                "Show all calculation steps. Give the final numerical answer."
            )
        
        solve_prompts.append((sys_msg, user_msg))

    # ===== BATCH SOLVE =====
    t1 = time.time()
    responses = batch_generate(solve_prompts, max_tokens=PHYSICS_MAX_TOKENS)
    t1_done = time.time() - t1
    print(f"  vLLM batch done in {t1_done:.1f}s ({t1_done/N:.2f}s/sample)")

    # ===== PARSE RESPONSES =====
    for i, raw in enumerate(responses):
        ans_data = safe_json(raw)
        predicted = str(ans_data.get("answer", "Unknown")).strip()
        
        # Build reasoning from steps or reasoning field
        reasoning = ans_data.get("reasoning", "")
        if not reasoning:
            steps = ans_data.get("steps", [])
            if steps:
                reasoning = " → ".join(steps)
        
        results[i].predicted_answers.append({
            "question_id": 0,
            "answer": predicted,
            "reasoning": reasoning,
            "predicted_unit": ans_data.get("unit", ""),
        })
        
        # Z3 not used for physics
        results[i].z3_status = "skipped"
        results[i].z3_compiled = 0
        results[i].z3_total = 0
        results[i].status = "success"

    # ===== FLEXIBLE ANSWER MATCHING =====
    match_count = 0
    for i, r in enumerate(results):
        gt = samples[i].get("answers", [])
        if not gt or not r.predicted_answers:
            continue
        
        pred = r.predicted_answers[0]["answer"]
        truth = gt[0]
        
        if _physics_answer_match(pred, truth):
            r.correct_count = 1
            match_count += 1
        else:
            r.correct_count = 0
        
        r.time_sec = round(time.time() - t0_all, 2)

    total_time = time.time() - t0_all
    print(f"\n  Pipeline {dataset_name} done in {total_time:.1f}s")
    print(f"  Avg: {total_time / N:.2f}s/sample")
    print(f"  Quick match check: {match_count}/{N} ({match_count/N*100:.1f}%)")

    return results


# ===================================================================
# MAIN ORCHESTRATOR
# ===================================================================

def run_batch_pipeline(samples, dataset_name="Dataset"):
    """
    Chay toan bo pipeline bang BATCH INFERENCE.

    Routing:
      - Physics samples (has _is_physics flag) -> Direct solver (skip FOL/Z3)
      - Logic samples  -> FOL formalization + Z3 verify + correction loop
    """
    N = len(samples)
    t0_all = time.time()

    results = [
        PipelineResult(
            sample_id=i,
            ground_truth=samples[i].get("answers", []),
            total_questions=len(samples[i].get("questions", [])),
        )
        for i in range(N)
    ]

    # ===== ROUTING: Physics vs Logic =====
    is_physics = any(s.get("_is_physics", False) for s in samples)

    if is_physics and PHYSICS_MODE == "direct":
        return _run_physics_pipeline(samples, results, N, t0_all, dataset_name)

    # ===== LOGIC PATH (UNCHANGED) =====

    # ===== ROUND 1: Batch Formalization =====
    print(f"\n{'=' * 60}")
    print(f"  [Round 1] Batch Formalization -- {N} samples")
    print(f"{'=' * 60}")

    t1 = time.time()
    form_prompts = [
        (FORMALIZATION_SYSTEM, _make_formalization_user(s))
        for s in samples
    ]
    form_responses = batch_generate(form_prompts, max_tokens=MAX_NEW_TOKENS)
    t1_done = time.time() - t1
    print(f"  vLLM batch done in {t1_done:.1f}s ({t1_done/N:.1f}s/sample)")

    for i, raw in enumerate(form_responses):
        results[i].z3_attempts = 1
        try:
            _process_formalization(results[i], raw)
        except Exception as e:
            results[i].z3_status = "no_ast"
            results[i].error_log.append(f"Round1: {str(e)[:300]}")

    # Stats
    status_counts = {}
    for r in results:
        status_counts[r.z3_status] = status_counts.get(r.z3_status, 0) + 1
    print(f"  Z3 results: {status_counts}")

    # ===== ROUNDS 2..MAX_RETRIES: Batch Correction =====
    for attempt in range(2, MAX_RETRIES + 1):
        # Chi correction cho compile_error (khong phai no_ast)
        failed = [i for i in range(N) if results[i].z3_status == "compile_error"]
        if not failed:
            print(f"  [Round {attempt}] Khong con loi compile -> skip")
            break

        print(f"\n  [Round {attempt}] Batch Correction -- {len(failed)} failed samples")

        t_c = time.time()
        corr_prompts = [
            (CORRECTION_SYSTEM, _make_correction_user(samples[i], results[i]))
            for i in failed
        ]
        corr_responses = batch_generate(corr_prompts, max_tokens=MAX_NEW_TOKENS)
        t_c_done = time.time() - t_c
        print(f"  vLLM batch done in {t_c_done:.1f}s")

        for j, raw in enumerate(corr_responses):
            idx = failed[j]
            results[idx].z3_attempts = attempt
            try:
                _process_formalization(results[idx], raw)
            except Exception as e:
                results[idx].error_log.append(f"Round{attempt}: {str(e)[:300]}")

        # Updated stats
        status_counts = {}
        for r in results:
            status_counts[r.z3_status] = status_counts.get(r.z3_status, 0) + 1
        print(f"  Z3 results: {status_counts}")

    # ===== BATCH ANSWER EXTRACTION =====
    print(f"\n{'=' * 60}")
    print(f"  [Answers] Batch Answer Extraction")
    print(f"{'=' * 60}")

    ans_prompts = []
    ans_mapping = []  # (sample_idx, question_idx)

    for i, s in enumerate(samples):
        fol_ctx = []
        for item in results[i].premises_ast:
            fol_ctx.append(item.get("source_nl", ""))
        for q_idx, q in enumerate(s.get("questions", [])):
            prompt = (ANSWER_SYSTEM, _make_answer_user(s, fol_ctx, q, q_idx))
            ans_prompts.append(prompt)
            ans_mapping.append((i, q_idx))

    if ans_prompts:
        t_a = time.time()
        ans_responses = batch_generate(ans_prompts, max_tokens=ANS_MAX_TOKENS)
        t_a_done = time.time() - t_a
        print(f"  vLLM batch done in {t_a_done:.1f}s ({len(ans_prompts)} questions)")

        for j, raw in enumerate(ans_responses):
            sample_idx, q_idx = ans_mapping[j]
            try:
                ans = safe_json(raw)
                results[sample_idx].predicted_answers.append({
                    "question_id": q_idx,
                    "answer": ans.get("answer", "Unknown"),
                    "reasoning": ans.get("reasoning", ""),
                })
            except Exception:
                results[sample_idx].predicted_answers.append({
                    "question_id": q_idx,
                    "answer": "Unknown",
                    "reasoning": raw[:200],
                })

    # ===== FINALIZE STATUS =====
    for i, r in enumerate(results):
        gt = samples[i].get("answers", [])
        correct = sum(
            1
            for ar in r.predicted_answers
            if ar["question_id"] < len(gt)
            and str(ar["answer"]).strip().upper()
            == str(gt[ar["question_id"]]).strip().upper()
        )
        r.correct_count = correct

        if r.z3_status in ("sat", "unsat", "unknown"):
            r.status = "success"
        elif r.z3_compiled > 0:
            r.status = "partial"
        else:
            r.status = "failed"

        r.time_sec = round(time.time() - t0_all, 2)

    total_time = time.time() - t0_all
    print(f"\n  Pipeline {dataset_name} hoan tat in {total_time:.1f}s")
    print(f"  Trung binh: {total_time / N:.1f}s/sample")

    return results


print("Batch pipeline orchestrator san sang (Logic + Physics)")


''')

# ================================================================
# CELL 8: Stage 5 -- Evaluation & Export (add "skipped" status)
# ================================================================
cells[8] = make_cell(r'''
# ==================================================================
# STAGE 5 -- Evaluation & Export
# ==================================================================

def evaluate(results):
    n = len(results)
    if n == 0:
        return {}
    total_q = sum(r.total_questions for r in results)
    total_ok = sum(r.correct_count for r in results)
    status_ct = {"success": 0, "partial": 0, "failed": 0}
    z3_ct = {"sat": 0, "unsat": 0, "unknown": 0, "compile_error": 0,
             "solver_error": 0, "no_ast": 0, "skipped": 0, "other": 0}
    for r in results:
        status_ct[r.status] = status_ct.get(r.status, 0) + 1
        key = r.z3_status if r.z3_status in z3_ct else "other"
        z3_ct[key] += 1
    hall_total = sum(len(r.hallucination_warn) for r in results)
    avg_retries = sum(r.z3_attempts for r in results) / n
    return {
        "n_samples": n,
        "total_questions": total_q,
        "total_correct": total_ok,
        "accuracy": round(total_ok / total_q, 4) if total_q else 0,
        "status_breakdown": status_ct,
        "z3_breakdown": z3_ct,
        "hallucination_warnings": hall_total,
        "avg_z3_retries": round(avg_retries, 2),
    }


def result_to_dict(r):
    return {
        "sample_id": r.sample_id,
        "status": r.status,
        "z3_status": r.z3_status,
        "z3_compiled": r.z3_compiled,
        "z3_total": r.z3_total,
        "z3_attempts": r.z3_attempts,
        "z3_errors": r.z3_errors[:3],
        "hallucination_warns": r.hallucination_warn,
        "local_ontology": r.local_ontology,
        "correct_count": r.correct_count,
        "total_questions": r.total_questions,
        "predicted_answers": [a["answer"] for a in r.predicted_answers],
        "ground_truth": r.ground_truth,
        "per_question": [
            {
                "q_id": a["question_id"],
                "predicted": a["answer"],
                "gt": r.ground_truth[a["question_id"]] if a["question_id"] < len(r.ground_truth) else "?",
                "correct": (
                    str(a["answer"]).upper() == str(r.ground_truth[a["question_id"]]).upper()
                    if a["question_id"] < len(r.ground_truth) else False
                ),
                "reasoning": a.get("reasoning", ""),
            }
            for a in r.predicted_answers
        ],
        "time_sec": r.time_sec,
        "error_log": r.error_log[-2:],
    }


def finalize_and_save(results, output_path, dataset_path, dataset_name="Dataset"):
    """Print summary va luu JSON."""
    if not results:
        print(f"[WARN] No results for {dataset_name}")
        return

    metrics = evaluate(results)
    acc_val = metrics.get("accuracy", 0)

    W = 60
    print("\n" + "=" * W)
    print(f"  {dataset_name.upper()} -- EVALUATION SUMMARY")
    print("=" * W)
    print(f"  Model          : {QWEN_MODEL_ID}")
    print(f"  Engine         : vLLM (TP={TENSOR_PARALLEL})")
    print(f"  Samples        : {metrics.get('n_samples', 0)}")
    print(f"  Total questions: {metrics.get('total_questions', 0)}")
    print(f"  Correct        : {metrics.get('total_correct', 0)}")
    print(f"  Accuracy       : {acc_val:.1%}")
    print("-" * W)
    print("  Pipeline Status:")
    for k, v in metrics.get("status_breakdown", {}).items():
        if v > 0:
            print(f"    {k:14}: {v:3d}  {'#' * min(v, 50)}")
    print("-" * W)
    print("  Z3 Verification:")
    for k, v in metrics.get("z3_breakdown", {}).items():
        if v > 0:
            print(f"    {k:16}: {v:3d}  {'#' * min(v, 50)}")
    print("-" * W)
    print(f"  Hallucination warns: {metrics.get('hallucination_warnings', 0)}")
    print(f"  Avg Z3 retries     : {metrics.get('avg_z3_retries', 0)}")
    print("=" * W)

    # Per-sample table (show first 50 + summary)
    hdr = f"{'ID':>3} | {'Status':>8} | {'Z3':>13} | {'Corr':>6} | {'Retry':>5} | Hall"
    print(hdr)
    print("-" * len(hdr))
    show_n = min(len(results), 50)
    for r in results[:show_n]:
        hall = f"W{len(r.hallucination_warn)}" if r.hallucination_warn else "ok"
        print(
            f"{r.sample_id:>3} | {r.status:>8} | {r.z3_status:>13} | "
            f"{r.correct_count}/{r.total_questions:>4} | {r.z3_attempts:>5} | {hall}"
        )
    if len(results) > show_n:
        print(f"  ... ({len(results) - show_n} more rows)")

    # Save JSON
    output_data = {
        "meta": {
            "model": QWEN_MODEL_ID,
            "engine": "vLLM",
            "tensor_parallel": TENSOR_PARALLEL,
            "n_samples": len(results),
            "max_retries": MAX_RETRIES,
            "dataset": dataset_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": metrics,
        "per_sample": [result_to_dict(r) for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    fsize = Path(output_path).stat().st_size / 1024
    total_c = metrics.get("total_correct", 0)
    total_q = metrics.get("total_questions", 0)
    print(f"\nKet qua luu tai: {output_path}")
    print(f"  Dung luong: {fsize:.1f} KB")
    print(f"  Final Accuracy: {acc_val:.1%}  ({total_c}/{total_q})")


# ==================================================================
# RUN
# ==================================================================

print("\n" + "=" * 65)
print("  NEURO-SYMBOLIC PIPELINE v2 -- vLLM BATCH INFERENCE")
print(f"  Model: {QWEN_MODEL_ID}  |  TP: {TENSOR_PARALLEL} GPU(s)")
print(f"  Physics Mode: {PHYSICS_MODE}")
print("=" * 65)

# --- Logic Dataset ---
logic_samples = load_dataset(DATASET_PATH, is_physics=False)
print(f"\nLogic Dataset: {len(logic_samples)} samples")

if logic_samples:
    logic_results = run_batch_pipeline(logic_samples, dataset_name="Logic Dataset")
    finalize_and_save(logic_results, OUTPUT_PATH, DATASET_PATH, "Logic Dataset")

# --- Physics Dataset ---
if PHYSICS_DATASET_PATH:
    physics_samples = load_dataset(PHYSICS_DATASET_PATH, is_physics=True)
    print(f"\nPhysics Dataset: {len(physics_samples)} samples")

    if physics_samples:
        physics_output = OUTPUT_PATH.replace(".json", "_physics.json")
        physics_results = run_batch_pipeline(physics_samples, dataset_name="Physics Dataset")
        finalize_and_save(physics_results, physics_output, PHYSICS_DATASET_PATH, "Physics Dataset")

print("\n" + "=" * 65)
print("  PIPELINE V2 HOAN TAT")
print("=" * 65)

''')

# ================================================================
# CLEAR ALL OUTPUTS from all cells
# ================================================================
for c in cells:
    c['outputs'] = []
    c['execution_count'] = None

# ================================================================
# WRITE NEW NOTEBOOK
# ================================================================
nb['cells'] = cells

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

fsize = len(json.dumps(nb, ensure_ascii=False))
print(f"\nWrote {OUT}")
print(f"  Cells: {len(cells)}")
print(f"  Size: {fsize / 1024:.1f} KB")
print("Done!")
