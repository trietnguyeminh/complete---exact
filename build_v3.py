#!/usr/bin/env python3
"""
Build notebook_v3_entailment.ipynb
Adds Z3 entailment-based answer derivation to the pipeline.
"""
import json, sys, copy
sys.stdout.reconfigure(encoding='utf-8')

ORIG = r'C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v2_physics.ipynb'
OUT  = r'C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v3_entailment.ipynb'

with open(ORIG, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Original: {len(cells)} cells")

def make_cell(source):
    return {
        "cell_type": "code",
        "source": source,
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None
    }

# ================================================================
# CELL 0: Update docstring
# ================================================================
cells[0] = make_cell('''#!/usr/bin/env python3
"""
vllm_pipeline_v3.py -- Neuro-Symbolic Pipeline with Z3 Entailment

EXACT 2026 -- XAI Challenge @ IJCNN
Qwen2.5-7B + Z3 + vLLM | Kaggle T4x2, No Cloud API

Pipeline v3 -- Z3 Entailment Answer Derivation:
  Stage 0: Install & Config
  Stage 1: Load vLLM Engine
  Stage 2: Data Grounding + Dual-Layer Ontology + Prompts
  Stage 3: Z3 Compiler + Entailment Checker (deterministic, no AI)
  Stage 4: Batch Pipeline (Formalize -> Z3 -> Entailment -> Answers)
  Stage 5: Evaluation & Export

Key improvement over v2:
  - Qwen formalizes BOTH premises AND questions into FOL AST
  - Z3 derives answers via entailment checking (deterministic, no LLM)
  - Qwen answer extraction only used as fallback when Z3 can't decide
"""
''')

# ================================================================
# CELL 1: Kaggle T4 fix -- keep as-is
# ================================================================
cells[1]['outputs'] = []
cells[1]['execution_count'] = None

# ================================================================
# CELL 2: Stage 0 Install -- keep as-is
# ================================================================
cells[2]['outputs'] = []
cells[2]['execution_count'] = None

# ================================================================
# CELL 3: CONFIG -- same as v2
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
TENSOR_PARALLEL  = min(N_GPUS, 2)
MAX_MODEL_LEN    = 8192
GPU_MEMORY_UTIL  = 0.85
DTYPE            = "half"

# Physics Config
PHYSICS_MODE       = "direct"
PHYSICS_MAX_TOKENS = 1024
PHYSICS_TOLERANCE  = 0.05

# Z3 Entailment Config
Z3_ENTAILMENT      = True          # True = dung Z3 entailment, False = dung Qwen
Z3_SOLVER_TIMEOUT  = 5000          # ms timeout cho Z3 solver
# ==================================================================

print(f"Config OK")
print(f"  Model         : {QWEN_MODEL_ID}")
print(f"  Tensor Parallel: {TENSOR_PARALLEL} GPU(s)")
print(f"  GPU Mem Util  : {GPU_MEMORY_UTIL}")
print(f"  Z3 Entailment : {Z3_ENTAILMENT}")
print(f"  Physics Mode  : {PHYSICS_MODE}")

""")

# ================================================================
# CELL 4: Stage 1 Load vLLM -- keep as-is
# ================================================================
cells[4]['outputs'] = []
cells[4]['execution_count'] = None

# ================================================================
# CELL 5: Stage 2 -- Prompts, Ontology, Helpers
# Now includes question formalization prompt with few-shot
# ================================================================
cells[5] = make_cell(r'''
# ==================================================================
# STAGE 2 -- Dual-Layer Ontology, Prompts, Data Loading
# ==================================================================

GLOBAL_ONTOLOGY = {
    "quantifiers": ["forall", "exists"],
    "logical_operators": ["and", "or", "implies", "iff", "not"],
    "ast_node_types": ["quantifier", "connective", "predicate", "variable", "constant"],
}

GLOBAL_ONTOLOGY_TEXT = """
## GLOBAL ONTOLOGY -- BAT BUOC TUAN THU

### Quantifiers:
  forall -> forall  |  exists -> exists

### Logical Operators:
  and, or, implies, iff, not

### AST Node Types (4 loai):
  quantifier : { "type":"quantifier",  "operator":"forall|exists",
                 "bound_variables":["x",...], "body":{...} }
  connective : { "type":"connective",  "operator":"and|or|implies|iff|not",
                 "operands":[{...},...] }
  predicate  : { "type":"predicate",   "name":"PredicateName",
                 "arguments":["x","y",...] }
  variable   : { "type":"variable",    "name":"x" }
  constant   : { "type":"constant",    "name":"SomeName" }

### QUY TAC:
  1. Chi dung 4 node types tren
  2. 'not' chi co DUNG 1 operand
  3. 'implies' co DUNG 2 operands
  4. bound_variables phai la list
  5. Variables: lowercase (x,y,z), Constants: PascalCase
"""

# ==================================================================
# FEW-SHOT EXAMPLE (with question formalization for entailment)
# ==================================================================

FEW_SHOT_EXAMPLE = """
### WORKED EXAMPLE:

Premises:
  P1: "All students who study hard pass the exam."
  P2: "Alice is a student."
  P3: "Alice studies hard."

Question: "Is it true that Alice passes the exam?"

Correct output:
{
  "step1_local_ontology": [
    {"predicate": "Student",   "arity": 1, "description": "x is a student"},
    {"predicate": "StudyHard", "arity": 1, "description": "x studies hard"},
    {"predicate": "PassExam",  "arity": 1, "description": "x passes the exam"}
  ],
  "step2_premises_ast": [
    {
      "premise_id": 0,
      "source_nl": "All students who study hard pass the exam.",
      "ast": {
        "type": "quantifier", "operator": "forall",
        "bound_variables": ["x"],
        "body": {
          "type": "connective", "operator": "implies",
          "operands": [
            {"type": "connective", "operator": "and", "operands": [
              {"type": "predicate", "name": "Student",   "arguments": ["x"]},
              {"type": "predicate", "name": "StudyHard", "arguments": ["x"]}
            ]},
            {"type": "predicate", "name": "PassExam", "arguments": ["x"]}
          ]
        }
      }
    },
    { "premise_id": 1, "source_nl": "Alice is a student.",
      "ast": {"type": "predicate", "name": "Student", "arguments": ["Alice"]} },
    { "premise_id": 2, "source_nl": "Alice studies hard.",
      "ast": {"type": "predicate", "name": "StudyHard", "arguments": ["Alice"]} }
  ],
  "step3_question_fol": [
    {
      "question_id": 0,
      "question_type": "yes_no",
      "source_nl": "Alice passes the exam",
      "statement_ast": {"type": "predicate", "name": "PassExam", "arguments": ["Alice"]}
    }
  ]
}
"""

# ==================================================================
# LOGIC PROMPT TEMPLATES
# ==================================================================

FORMALIZATION_SYSTEM = (
    "Ban la chuyen gia logic hinh thuc (FOL). Nhiem vu:\n"
    "  Buoc 1: Tao LOCAL ONTOLOGY -- danh sach Predicates\n"
    "  Buoc 2: Chuyen TUNG tien de thanh cay AST JSON\n"
    "  Buoc 3: Chuyen TUNG cau hoi thanh AST JSON de Z3 kiem tra entailment\n\n"
    + GLOBAL_ONTOLOGY_TEXT + "\n"
    + FEW_SHOT_EXAMPLE + "\n"
    "QUAN TRONG ve Buoc 3 (question formalization):\n"
    "  - question_type: 'yes_no' hoac 'multiple_choice'\n"
    "  - yes_no: chuyen statement thanh 1 AST node (statement_ast)\n"
    "  - multiple_choice: chuyen TUNG option A/B/C/D thanh AST (options_ast)\n"
    "  - Dung CUNG predicates da khai bao trong step1\n"
    "  - CHI formalize LOGIC CONTENT, bo phan text thua\n\n"
    "Output JSON THUAN TUY (khong markdown, khong text thua):\n"
    '{\n'
    '  "step1_local_ontology": [\n'
    '    {"predicate": "Name", "arity": N, "description": "..."}\n'
    '  ],\n'
    '  "step2_premises_ast": [\n'
    '    {"premise_id": 0, "source_nl": "...", "ast": { <AST> }}\n'
    '  ],\n'
    '  "step3_question_fol": [\n'
    '    {\n'
    '      "question_id": 0,\n'
    '      "question_type": "yes_no",\n'
    '      "source_nl": "statement being checked",\n'
    '      "statement_ast": { <AST> }\n'
    '    }\n'
    '  ]\n'
    '}\n'
)

CORRECTION_SYSTEM = (
    "Ban la chuyen gia sua loi FOL. He thong Z3 da phat hien loi.\n"
    "Nhiem vu: sua lai TOAN BO (Buoc 1 + Buoc 2 + Buoc 3) de khong con loi.\n\n"
    + GLOBAL_ONTOLOGY_TEXT
    + "\nLoi hay gap:\n"
    "  - Arity khong nhat quan\n"
    "  - Variable chua khai bao trong bound_variables\n"
    "  - Predicate khong co trong Local Ontology\n"
    "  - 'not' co nhieu hon 1 operand\n"
    "  - 'implies' khong du 2 operands\n"
    "  - step3_question_fol thieu hoac sai format\n\n"
    "Output JSON thuan tuy -- format GIONG HET lan dau.\n"
)

# Fallback: khi Z3 entailment khong the xac dinh
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

# ===== PHYSICS PROMPTS (unchanged from v2) =====

PHYSICS_SOLVER_SYSTEM = (
    "You are an expert physics problem solver. "
    "Solve the given problem step-by-step with clear calculations.\n\n"
    "Rules:\n"
    "  - Identify all given quantities with correct units\n"
    "  - State the relevant physics formula(s)\n"
    "  - Show unit conversions explicitly\n"
    "  - Compute step by step\n"
    "  - Give the FINAL numerical answer\n\n"
    "Output PURE JSON (no markdown):\n"
    '{"steps": ["step1...", "step2..."], '
    '"answer": "<number>", "unit": "<unit>"}\n'
)

PHYSICS_MC_SYSTEM = (
    "You are an expert physics problem solver. "
    "Solve the given multiple-choice problem.\n\n"
    "Output PURE JSON (no markdown):\n"
    '{"reasoning": "...", "answer": "<answer>"}\n'
)

print("Prompt templates san sang (Logic v3 + Physics)")

# ==================================================================
# DATASET LOADER (same as v2)
# ==================================================================

def load_dataset(path, is_physics=False, max_samples=N_SAMPLES):
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
            s["premises-NL"] = []
            s["questions"]   = [s.get("question", "")]
            s["answers"]     = [str(s.get("answer", "Unknown"))]
            s["_unit"]       = s.get("unit", "")
            s["_cot"]        = s.get("cot", "")
            s["_is_physics"] = True
    return out

# ==================================================================
# JSON PARSER
# ==================================================================

def safe_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
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

# ==================================================================
# BATCH GENERATE
# ==================================================================

def batch_generate(prompt_pairs, max_tokens=MAX_NEW_TOKENS):
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
    outputs_sorted = sorted(outputs, key=lambda x: int(x.request_id))
    return [o.outputs[0].text.strip() for o in outputs_sorted]

print("batch_generate() san sang")

''')

# ================================================================
# CELL 6: Stage 3 -- Z3 Compiler + ENTAILMENT CHECKER
# ================================================================
cells[6] = make_cell(r'''
# ==================================================================
# STAGE 3 -- Z3 Compiler + Entailment Checker (deterministic, no AI)
# ==================================================================

_func_cache = {}


def get_z3_func(name, arity):
    key = f"{name}_{arity}"
    if key not in _func_cache:
        sorts = [IntSort()] * arity + [BoolSort()]
        _func_cache[key] = Function(name, *sorts)
    return _func_cache[key]


def _resolve_bound_var_name(bv):
    if isinstance(bv, dict):
        return bv.get("name", str(bv))
    return str(bv)


def _resolve_predicate_arg(a, var_map):
    if isinstance(a, str):
        if a in var_map:
            return var_map[a]
        return IntVal(abs(hash(a)) % 100000)
    if isinstance(a, dict):
        atype = a.get("type", "")
        name = a.get("name", "")
        if atype == "variable":
            if name in var_map:
                return var_map[name]
            v = Int(name)
            var_map[name] = v
            return v
        if atype == "constant":
            if name in var_map:
                return var_map[name]
            return IntVal(abs(hash(name)) % 100000)
        raise ValueError(f"Argument khong hop le (type={atype!r})")
    return IntVal(abs(hash(str(a))) % 100000)


def compile_ast(node, var_map):
    """Bien dich 1 AST node -> Z3 expression."""
    if not isinstance(node, dict):
        raise ValueError(f"Expected dict, got {type(node)}: {node!r}")

    ntype = node.get("type", "")

    if ntype == "quantifier":
        op = node.get("operator", "").lower()
        bvs = node.get("bound_variables", [])
        if not bvs:
            raise ValueError("quantifier thieu bound_variables")
        bv_names = [_resolve_bound_var_name(bv) for bv in bvs]
        z3_bvs = [Int(v) for v in bv_names]
        child_map = {**var_map, **{v: z3_bvs[i] for i, v in enumerate(bv_names)}}
        body = compile_ast(node["body"], child_map)
        if op == "forall":
            return ForAll(z3_bvs, body)
        elif op in ("exists", "exist"):
            return Exists(z3_bvs, body)
        else:
            raise ValueError(f"Quantifier khong hop le: {op!r}")

    elif ntype == "connective":
        op = node.get("operator", "").lower()
        ops = [compile_ast(o, var_map) for o in node.get("operands", [])]
        if op == "and":
            return And(*ops)
        elif op == "or":
            return Or(*ops)
        elif op == "implies":
            if len(ops) != 2:
                raise ValueError(f"implies can 2 operands, nhan {len(ops)}")
            return Implies(ops[0], ops[1])
        elif op == "iff":
            if len(ops) != 2:
                raise ValueError(f"iff can 2 operands, nhan {len(ops)}")
            return And(Implies(ops[0], ops[1]), Implies(ops[1], ops[0]))
        elif op == "not":
            if len(ops) != 1:
                raise ValueError(f"not can 1 operand, nhan {len(ops)}")
            return Not(ops[0])
        else:
            raise ValueError(f"Connective khong hop le: {op!r}")

    elif ntype == "predicate":
        name = node.get("name", "")
        args = node.get("arguments", [])
        if not name:
            raise ValueError('predicate thieu "name"')
        func = get_z3_func(name, len(args))
        z3_args = [_resolve_predicate_arg(a, var_map) for a in args]
        return func(*z3_args)

    elif ntype in ("variable", "constant"):
        name = node.get("name", "")
        if name in var_map:
            return var_map[name]
        if ntype == "constant":
            return IntVal(abs(hash(name)) % 100000)
        v = Int(name)
        var_map[name] = v
        return v

    else:
        raise ValueError(f"AST node type khong hop le: {ntype!r}")


def verify_with_z3(premises_ast):
    """Bien dich toan bo premises AST -> Z3, kiem tra consistency."""
    _func_cache.clear()
    solver = Solver()
    errors = []
    compiled = 0

    for item in premises_ast:
        pid = item.get("premise_id", "?")
        try:
            ast = item.get("ast", {})
            if not ast:
                errors.append(f"Premise {pid}: AST rong")
                continue
            expr = compile_ast(ast, {})
            solver.add(expr)
            compiled += 1
        except Exception as e:
            errors.append(f"Premise {pid}: {str(e)[:250]}")

    if errors:
        return {
            "status": "compile_error",
            "errors": errors,
            "compiled_count": compiled,
            "total_count": len(premises_ast),
        }
    try:
        result = solver.check()
        return {
            "status": str(result),
            "errors": [],
            "compiled_count": compiled,
            "total_count": len(premises_ast),
        }
    except Exception as e:
        return {
            "status": "solver_error",
            "errors": [str(e)],
            "compiled_count": compiled,
            "total_count": len(premises_ast),
        }


def hallucination_check(local_ontology, premises_ast):
    """Kiem tra: moi Predicate trong AST phai co trong Local Ontology."""
    declared = set()
    for item in local_ontology:
        if isinstance(item, dict):
            declared.add(item.get("predicate", ""))

    warnings = []

    def collect_predicates(node, found):
        if not isinstance(node, dict):
            return
        if node.get("type") == "predicate":
            found.add(node.get("name", ""))
        for v in node.values():
            if isinstance(v, dict):
                collect_predicates(v, found)
            elif isinstance(v, list):
                for sub in v:
                    collect_predicates(sub, found)

    for item in premises_ast:
        used = set()
        collect_predicates(item.get("ast", {}), used)
        hallucinated = used - declared - {""}
        if hallucinated:
            warnings.append(
                f"Premise {item.get('premise_id', '?')}: "
                f"predicates not in Ontology -> {hallucinated}"
            )
    return warnings


# ==================================================================
# Z3 ENTAILMENT CHECKER (NEW in v3)
# ==================================================================

def _compile_premises_to_solver(premises_ast):
    """Compile all premises into a Z3 solver. Return (solver, compiled, errors)."""
    _func_cache.clear()
    solver = Solver()
    solver.set("timeout", Z3_SOLVER_TIMEOUT)
    errors = []
    compiled = 0

    for item in premises_ast:
        pid = item.get("premise_id", "?")
        try:
            ast_node = item.get("ast", {})
            if not ast_node:
                continue
            expr = compile_ast(ast_node, {})
            solver.add(expr)
            compiled += 1
        except Exception as e:
            errors.append(f"P{pid}: {str(e)[:200]}")

    return solver, compiled, errors


def z3_entailment_check(premises_ast, question_fol_item):
    """
    Z3 entailment-based answer derivation for ONE question.

    Yes/No:
      YES:     premises + NOT(stmt) -> UNSAT  (stmt is entailed)
      NO:      premises + stmt -> UNSAT       (stmt contradicts)
      UNKNOWN: both SAT                       (undetermined)

    MC:
      Test each option for entailment, return the entailed one.

    Returns: {"answer": "...", "method": "...", "detail": "..."}
    """
    q_type = question_fol_item.get("question_type", "yes_no")

    if q_type == "yes_no":
        return _entail_yes_no(premises_ast, question_fol_item)
    elif q_type == "multiple_choice":
        return _entail_mc(premises_ast, question_fol_item)
    else:
        return {"answer": "Unknown", "method": "unsupported_qtype"}


def _entail_yes_no(premises_ast, q_item):
    """Entailment check for Yes/No/Unknown questions."""
    stmt_ast = q_item.get("statement_ast", {})
    if not stmt_ast:
        return {"answer": "Unknown", "method": "no_statement_ast"}

    # Compile the statement
    try:
        _func_cache_backup = dict(_func_cache)
        stmt_expr = compile_ast(stmt_ast, {})
    except Exception as e:
        return {"answer": "Unknown", "method": "stmt_compile_error",
                "detail": str(e)[:200]}

    # --- Test YES: premises + NOT(stmt) -> UNSAT? ---
    solver1, c1, e1 = _compile_premises_to_solver(premises_ast)
    if e1:
        return {"answer": "Unknown", "method": "premise_compile_error",
                "detail": "; ".join(e1[:3])}

    try:
        solver1.push()
        solver1.add(Not(stmt_expr))
        r1 = solver1.check()
        solver1.pop()
    except Exception as e:
        r1 = None

    if r1 == unsat:
        return {"answer": "Yes", "method": "z3_entailment",
                "detail": "premises + NOT(Q) is UNSAT => Q is entailed by premises"}

    # --- Test NO: premises + stmt -> UNSAT? ---
    solver2, _, _ = _compile_premises_to_solver(premises_ast)
    try:
        solver2.push()
        solver2.add(stmt_expr)
        r2 = solver2.check()
        solver2.pop()
    except Exception as e:
        r2 = None

    if r2 == unsat:
        return {"answer": "No", "method": "z3_negation",
                "detail": "premises + Q is UNSAT => Q contradicts premises"}

    # --- Neither -> Unknown ---
    return {"answer": "Unknown", "method": "z3_undetermined",
            "detail": "Neither Q nor NOT(Q) entailed"}


def _entail_mc(premises_ast, q_item):
    """Entailment check for Multiple Choice (A/B/C/D)."""
    options_ast = q_item.get("options_ast", {})
    if not options_ast:
        return {"answer": "Unknown", "method": "no_options_ast"}

    entailed = []
    consistent = []

    for label in ("A", "B", "C", "D"):
        opt_ast = options_ast.get(label, {})
        if not opt_ast:
            continue

        try:
            opt_expr = compile_ast(opt_ast, {})
        except:
            continue

        # Entailment: premises + NOT(option) -> UNSAT?
        solver, c, e = _compile_premises_to_solver(premises_ast)
        if e:
            continue

        try:
            solver.push()
            solver.add(Not(opt_expr))
            r = solver.check()
            solver.pop()

            if r == unsat:
                entailed.append(label)

            # Consistency: premises + option -> SAT?
            solver.push()
            solver.add(opt_expr)
            r2 = solver.check()
            solver.pop()
            if r2 == sat:
                consistent.append(label)
        except:
            continue

    if len(entailed) == 1:
        return {"answer": entailed[0], "method": "z3_unique_entailment",
                "detail": f"Only option {entailed[0]} is entailed"}
    elif len(entailed) > 1:
        return {"answer": entailed[0], "method": "z3_multi_entailment",
                "detail": f"Options {entailed} all entailed, picking first"}
    elif len(consistent) == 1:
        return {"answer": consistent[0], "method": "z3_unique_consistent",
                "detail": f"Only option {consistent[0]} is consistent with premises"}
    else:
        return {"answer": "Unknown", "method": "z3_mc_undetermined",
                "detail": f"Entailed={entailed}, Consistent={consistent}"}


print("Z3 compiler + entailment checker san sang")

''')

# ================================================================
# CELL 7: Stage 4 -- Pipeline Orchestrator with Entailment
# ================================================================
cells[7] = make_cell(r'''
# ==================================================================
# STAGE 4 -- Batch Pipeline with Z3 Entailment
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
    question_fol: list = field(default_factory=list)     # NEW: formalized questions
    predicted_answers: list = field(default_factory=list)
    ground_truth: list = field(default_factory=list)
    total_questions: int = 0
    correct_count: int = 0
    time_sec: float = 0.0
    error_log: list = field(default_factory=list)
    answer_source: list = field(default_factory=list)    # NEW: "z3" or "qwen" per question


# ===================================================================
# PROMPT BUILDERS
# ===================================================================

def _make_formalization_user(sample):
    """Tao user prompt: premises + questions for FOL formalization."""
    premises = sample["premises-NL"]
    questions = sample.get("questions", [])

    numbered_p = "\n".join(f"Premise {i+1}: {p}" for i, p in enumerate(premises))
    numbered_q = "\n".join(f"Question {i+1}: {q}" for i, q in enumerate(questions))

    return (
        "Hay hinh thuc hoa cac tien de VA cau hoi sau theo quy trinh 3 buoc.\n\n"
        "=== PREMISES ===\n"
        + numbered_p + "\n\n"
        "=== QUESTIONS ===\n"
        + numbered_q + "\n\n"
        "Nho:\n"
        "  Buoc 1: khai bao Local Ontology\n"
        "  Buoc 2: sinh AST cho TUNG premise\n"
        "  Buoc 3: sinh AST cho TUNG question (de Z3 check entailment)\n"
        "    - yes_no: trich statement va chuyen thanh statement_ast\n"
        "    - multiple_choice (co A/B/C/D): chuyen tung option thanh options_ast\n"
        "  Chi tra ve JSON thuan tuy."
    )


def _make_correction_user(sample, result):
    """Correction prompt khi Z3 bao loi."""
    premises = sample["premises-NL"]
    questions = sample.get("questions", [])
    numbered_p = "\n".join(f"Premise {i+1}: {p}" for i, p in enumerate(premises))
    numbered_q = "\n".join(f"Question {i+1}: {q}" for i, q in enumerate(questions))
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
        f"{numbered_p}\n\n"
        "===== QUESTIONS ====\n"
        f"{numbered_q}\n\n"
        "Sua lai TOAN BO (Buoc 1 + Buoc 2 + Buoc 3). Chi tra ve JSON thuan tuy."
    )


def _make_answer_user(sample, fol_context, question, q_idx):
    """Fallback: Qwen answer extraction (khi Z3 entailment undetermined)."""
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


# ===================================================================
# PROCESS FORMALIZATION (updated for step3)
# ===================================================================

def _process_formalization(result, raw_response):
    """Parse formalization response (3 steps) va chay Z3."""
    formalization = safe_json(raw_response)
    local_onto = formalization.get("step1_local_ontology", [])
    premises_ast = formalization.get("step2_premises_ast", [])
    question_fol = formalization.get("step3_question_fol", [])

    result.local_ontology = local_onto
    result.premises_ast = premises_ast
    result.question_fol = question_fol

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
# PHYSICS PIPELINE (unchanged from v2)
# ===================================================================

def _normalize_number(s):
    s = str(s).strip()
    s = re.sub(r'\s*[A-Za-zμΩ°/%%]+\s*$', '', s).strip()
    m = re.match(
        r'^([+-]?[\d.]+)\s*[×xX\*]\s*10\s*\^?\s*\(?\s*([+-]?\d+)\s*\)?\s*$', s)
    if m:
        try:
            return float(m.group(1)) * (10 ** int(m.group(2)))
        except:
            pass
    try:
        return float(s)
    except:
        pass
    return None


def _physics_answer_match(predicted, ground_truth, tolerance=None):
    if tolerance is None:
        tolerance = PHYSICS_TOLERANCE
    p = str(predicted).strip()
    g = str(ground_truth).strip()
    if p.lower() == g.lower():
        return True
    p_num = _normalize_number(p)
    g_num = _normalize_number(g)
    if p_num is not None and g_num is not None:
        if g_num == 0:
            return abs(p_num) < 1e-9
        return abs(p_num - g_num) / abs(g_num) <= tolerance
    p_low, g_low = p.lower(), g.lower()
    if len(g_low) > 3 and (g_low in p_low or p_low in g_low):
        return True
    p_c = re.sub(r'[^\d.eE+\-×xX\*\^()]+', '', p)
    g_c = re.sub(r'[^\d.eE+\-×xX\*\^()]+', '', g)
    if p_c and g_c:
        p2, g2 = _normalize_number(p_c), _normalize_number(g_c)
        if p2 is not None and g2 is not None and g2 != 0:
            if abs(p2 - g2) / abs(g2) <= tolerance:
                return True
    return False


def _run_physics_pipeline(samples, results, N, t0_all, dataset_name):
    print(f"\n{'=' * 60}")
    print(f"  [Physics Mode] Direct Solve -- {N} samples")
    print(f"{'=' * 60}")
    solve_prompts = []
    for s in samples:
        question = s["questions"][0]
        unit = s.get("_unit", "")
        is_mc = any(opt in question for opt in ["A.", "B.", "C.", "D.", "(A)", "(B)"])
        if is_mc:
            solve_prompts.append((PHYSICS_MC_SYSTEM,
                f"Problem:\n{question}\n\nSolve and select the correct answer."))
        else:
            uh = f"\nExpected unit: {unit}" if unit and unit != "—" else ""
            solve_prompts.append((PHYSICS_SOLVER_SYSTEM,
                f"Problem:\n{question}{uh}\n\nSolve step-by-step."))
    t1 = time.time()
    responses = batch_generate(solve_prompts, max_tokens=PHYSICS_MAX_TOKENS)
    print(f"  vLLM done in {time.time()-t1:.1f}s ({(time.time()-t1)/N:.2f}s/sample)")
    mc = 0
    for i, raw in enumerate(responses):
        ad = safe_json(raw)
        pred = str(ad.get("answer", "Unknown")).strip()
        rsn = ad.get("reasoning", " | ".join(ad.get("steps", [])))
        results[i].predicted_answers.append({"question_id": 0, "answer": pred, "reasoning": rsn})
        results[i].z3_status = "skipped"
        results[i].status = "success"
        gt = samples[i].get("answers", [])
        if gt and _physics_answer_match(pred, gt[0]):
            results[i].correct_count = 1
            mc += 1
        results[i].time_sec = round(time.time() - t0_all, 2)
    print(f"  Quick match: {mc}/{N} ({mc/N*100:.1f}%)")
    return results


# ===================================================================
# MAIN PIPELINE ORCHESTRATOR
# ===================================================================

def run_batch_pipeline(samples, dataset_name="Dataset"):
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

    # ===== ROUTING =====
    is_physics = any(s.get("_is_physics", False) for s in samples)
    if is_physics and PHYSICS_MODE == "direct":
        return _run_physics_pipeline(samples, results, N, t0_all, dataset_name)

    # ===== LOGIC PATH =====

    # --- ROUND 1: Batch Formalization (premises + questions) ---
    print(f"\n{'=' * 60}")
    print(f"  [Round 1] Batch Formalization -- {N} samples")
    print(f"  (Formalizing premises + questions for Z3 entailment)")
    print(f"{'=' * 60}")

    t1 = time.time()
    form_prompts = [(FORMALIZATION_SYSTEM, _make_formalization_user(s)) for s in samples]
    form_responses = batch_generate(form_prompts, max_tokens=MAX_NEW_TOKENS)
    print(f"  vLLM done in {time.time()-t1:.1f}s ({(time.time()-t1)/N:.1f}s/sample)")

    for i, raw in enumerate(form_responses):
        results[i].z3_attempts = 1
        try:
            _process_formalization(results[i], raw)
        except Exception as e:
            results[i].z3_status = "no_ast"
            results[i].error_log.append(f"Round1: {str(e)[:300]}")

    status_counts = {}
    for r in results:
        status_counts[r.z3_status] = status_counts.get(r.z3_status, 0) + 1
    print(f"  Z3 results: {status_counts}")

    q_fol_count = sum(1 for r in results if r.question_fol)
    print(f"  Questions formalized: {q_fol_count}/{N} samples have step3")

    # --- ROUNDS 2..MAX_RETRIES: Batch Correction ---
    for attempt in range(2, MAX_RETRIES + 1):
        failed = [i for i in range(N) if results[i].z3_status == "compile_error"]
        if not failed:
            print(f"  [Round {attempt}] No compile errors -> skip")
            break

        print(f"\n  [Round {attempt}] Batch Correction -- {len(failed)} failed")
        t_c = time.time()
        corr_prompts = [(CORRECTION_SYSTEM, _make_correction_user(samples[i], results[i]))
                        for i in failed]
        corr_responses = batch_generate(corr_prompts, max_tokens=MAX_NEW_TOKENS)
        print(f"  vLLM done in {time.time()-t_c:.1f}s")

        for j, raw in enumerate(corr_responses):
            idx = failed[j]
            results[idx].z3_attempts = attempt
            try:
                _process_formalization(results[idx], raw)
            except Exception as e:
                results[idx].error_log.append(f"Round{attempt}: {str(e)[:300]}")

        status_counts = {}
        for r in results:
            status_counts[r.z3_status] = status_counts.get(r.z3_status, 0) + 1
        print(f"  Z3 results: {status_counts}")

    # ===== Z3 ENTAILMENT ANSWER DERIVATION =====
    print(f"\n{'=' * 60}")
    if Z3_ENTAILMENT:
        print(f"  [Answers] Z3 Entailment + Qwen Fallback")
    else:
        print(f"  [Answers] Qwen Answer Extraction (entailment disabled)")
    print(f"{'=' * 60}")

    z3_answered = 0
    z3_yes = 0
    z3_no = 0
    z3_unknown = 0
    qwen_fallback_items = []  # (sample_idx, q_idx, question_text)

    if Z3_ENTAILMENT:
        for i, s in enumerate(samples):
            r = results[i]
            questions = s.get("questions", [])

            # Z3 entailment only for compiled samples with question FOL
            can_entail = (
                r.z3_status in ("sat", "unsat", "unknown")
                and r.question_fol
                and len(r.question_fol) > 0
            )

            if can_entail:
                for q_idx, q in enumerate(questions):
                    # Find matching question_fol item
                    q_fol_item = None
                    for qf in r.question_fol:
                        if isinstance(qf, dict) and qf.get("question_id", -1) == q_idx:
                            q_fol_item = qf
                            break
                    # Fallback: use first/only question_fol if no ID match
                    if q_fol_item is None and len(r.question_fol) > q_idx:
                        q_fol_item = r.question_fol[q_idx]

                    if q_fol_item and isinstance(q_fol_item, dict):
                        try:
                            z3_result = z3_entailment_check(r.premises_ast, q_fol_item)
                            answer = z3_result.get("answer", "Unknown")
                            method = z3_result.get("method", "unknown")

                            r.predicted_answers.append({
                                "question_id": q_idx,
                                "answer": answer,
                                "reasoning": f"[Z3:{method}] {z3_result.get('detail','')}",
                            })
                            r.answer_source.append("z3")
                            z3_answered += 1

                            if answer == "Yes": z3_yes += 1
                            elif answer == "No": z3_no += 1
                            else: z3_unknown += 1

                        except Exception as e:
                            # Z3 error -> fallback to Qwen
                            qwen_fallback_items.append((i, q_idx, q))
                            r.answer_source.append("qwen")
                    else:
                        qwen_fallback_items.append((i, q_idx, q))
                        r.answer_source.append("qwen")
            else:
                # No Z3 -> all questions go to Qwen
                for q_idx, q in enumerate(questions):
                    qwen_fallback_items.append((i, q_idx, q))
                    r.answer_source.append("qwen")

        print(f"  Z3 entailment answers: {z3_answered} (Yes={z3_yes}, No={z3_no}, Unknown={z3_unknown})")
        print(f"  Qwen fallback needed:  {len(qwen_fallback_items)}")
    else:
        # No entailment: all go to Qwen
        for i, s in enumerate(samples):
            for q_idx, q in enumerate(s.get("questions", [])):
                qwen_fallback_items.append((i, q_idx, q))

    # ===== QWEN FALLBACK =====
    if qwen_fallback_items:
        print(f"\n  [Qwen Fallback] Batch extraction -- {len(qwen_fallback_items)} questions")

        ans_prompts = []
        for sample_idx, q_idx, q_text in qwen_fallback_items:
            fol_ctx = [item.get("source_nl", "") for item in results[sample_idx].premises_ast]
            prompt = (ANSWER_SYSTEM, _make_answer_user(samples[sample_idx], fol_ctx, q_text, q_idx))
            ans_prompts.append(prompt)

        t_a = time.time()
        ans_responses = batch_generate(ans_prompts, max_tokens=ANS_MAX_TOKENS)
        print(f"  vLLM done in {time.time()-t_a:.1f}s")

        for j, raw in enumerate(ans_responses):
            sample_idx, q_idx, _ = qwen_fallback_items[j]
            try:
                ans = safe_json(raw)
                results[sample_idx].predicted_answers.append({
                    "question_id": q_idx,
                    "answer": ans.get("answer", "Unknown"),
                    "reasoning": f"[Qwen] {ans.get('reasoning', '')}",
                })
            except Exception:
                results[sample_idx].predicted_answers.append({
                    "question_id": q_idx,
                    "answer": "Unknown",
                    "reasoning": f"[Qwen] {raw[:200]}",
                })

    # ===== FINALIZE =====
    for i, r in enumerate(results):
        gt = samples[i].get("answers", [])
        correct = sum(
            1 for ar in r.predicted_answers
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
    print(f"\n  Pipeline {dataset_name} done in {total_time:.1f}s")
    print(f"  Avg: {total_time / N:.1f}s/sample")

    return results


print("Pipeline orchestrator v3 san sang (Z3 Entailment + Qwen Fallback)")

''')

# ================================================================
# CELL 8: Stage 5 -- Evaluation (updated for answer source tracking)
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

    # Answer source breakdown
    z3_answers = sum(1 for r in results for s in r.answer_source if s == "z3")
    qwen_answers = sum(1 for r in results for s in r.answer_source if s == "qwen")

    # Z3 entailment accuracy (only Z3-answered questions)
    z3_correct = 0
    z3_total_q = 0
    qwen_correct = 0
    qwen_total_q = 0
    for r in results:
        gt = r.ground_truth
        for j, ar in enumerate(r.predicted_answers):
            qid = ar["question_id"]
            if qid >= len(gt):
                continue
            is_correct = str(ar["answer"]).strip().upper() == str(gt[qid]).strip().upper()
            src = r.answer_source[j] if j < len(r.answer_source) else "qwen"
            if src == "z3":
                z3_total_q += 1
                if is_correct:
                    z3_correct += 1
            else:
                qwen_total_q += 1
                if is_correct:
                    qwen_correct += 1

    return {
        "n_samples": n,
        "total_questions": total_q,
        "total_correct": total_ok,
        "accuracy": round(total_ok / total_q, 4) if total_q else 0,
        "status_breakdown": status_ct,
        "z3_breakdown": z3_ct,
        "hallucination_warnings": hall_total,
        "avg_z3_retries": round(avg_retries, 2),
        "answer_source": {
            "z3_entailment": z3_answers,
            "qwen_fallback": qwen_answers,
        },
        "z3_entailment_accuracy": round(z3_correct / z3_total_q, 4) if z3_total_q else 0,
        "z3_entailment_correct": z3_correct,
        "z3_entailment_total": z3_total_q,
        "qwen_accuracy": round(qwen_correct / qwen_total_q, 4) if qwen_total_q else 0,
        "qwen_correct": qwen_correct,
        "qwen_total": qwen_total_q,
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
        "answer_sources": r.answer_source,
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
                "source": r.answer_source[j] if j < len(r.answer_source) else "?",
            }
            for j, a in enumerate(r.predicted_answers)
        ],
        "time_sec": r.time_sec,
        "error_log": r.error_log[-2:],
    }


def finalize_and_save(results, output_path, dataset_path, dataset_name="Dataset"):
    if not results:
        print(f"[WARN] No results for {dataset_name}")
        return

    metrics = evaluate(results)
    acc_val = metrics.get("accuracy", 0)

    W = 65
    print("\n" + "=" * W)
    print(f"  {dataset_name.upper()} -- EVALUATION SUMMARY (v3)")
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

    # Answer source breakdown
    src = metrics.get("answer_source", {})
    z3_a = src.get("z3_entailment", 0)
    qw_a = src.get("qwen_fallback", 0)
    print(f"  Answer Sources:")
    print(f"    Z3 entailment : {z3_a:>4} questions")
    print(f"    Qwen fallback : {qw_a:>4} questions")
    if z3_a > 0:
        z3_acc = metrics.get("z3_entailment_accuracy", 0)
        z3_c = metrics.get("z3_entailment_correct", 0)
        z3_t = metrics.get("z3_entailment_total", 0)
        print(f"  Z3 Entailment Accuracy: {z3_acc:.1%} ({z3_c}/{z3_t})")
    if qw_a > 0:
        qw_acc = metrics.get("qwen_accuracy", 0)
        qw_c = metrics.get("qwen_correct", 0)
        qw_t = metrics.get("qwen_total", 0)
        print(f"  Qwen Fallback Accuracy: {qw_acc:.1%} ({qw_c}/{qw_t})")
    print("-" * W)
    print(f"  Hallucination warns: {metrics.get('hallucination_warnings', 0)}")
    print(f"  Avg Z3 retries     : {metrics.get('avg_z3_retries', 0)}")
    print("=" * W)

    # Per-sample table
    hdr = f"{'ID':>3} | {'Status':>8} | {'Z3':>13} | {'Corr':>6} | {'Src':>7} | Hall"
    print(hdr)
    print("-" * len(hdr))
    show_n = min(len(results), 50)
    for r in results[:show_n]:
        hall = f"W{len(r.hallucination_warn)}" if r.hallucination_warn else "ok"
        src_str = "/".join(r.answer_source[:2]) if r.answer_source else "?"
        print(
            f"{r.sample_id:>3} | {r.status:>8} | {r.z3_status:>13} | "
            f"{r.correct_count}/{r.total_questions:>4} | {src_str:>7} | {hall}"
        )
    if len(results) > show_n:
        print(f"  ... ({len(results) - show_n} more)")

    # Save
    output_data = {
        "meta": {
            "model": QWEN_MODEL_ID,
            "engine": "vLLM",
            "tensor_parallel": TENSOR_PARALLEL,
            "pipeline_version": "v3_entailment",
            "z3_entailment_enabled": Z3_ENTAILMENT,
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
    print(f"\nSaved: {output_path} ({fsize:.1f} KB)")
    print(f"Final Accuracy: {acc_val:.1%}  ({metrics.get('total_correct',0)}/{metrics.get('total_questions',0)})")


# ==================================================================
# RUN
# ==================================================================

print("\n" + "=" * 65)
print("  NEURO-SYMBOLIC PIPELINE v3 -- Z3 ENTAILMENT")
print(f"  Model: {QWEN_MODEL_ID}  |  TP: {TENSOR_PARALLEL} GPU(s)")
print(f"  Z3 Entailment: {Z3_ENTAILMENT}  |  Physics: {PHYSICS_MODE}")
print("=" * 65)

# --- Logic ---
logic_samples = load_dataset(DATASET_PATH, is_physics=False)
print(f"\nLogic Dataset: {len(logic_samples)} samples")
if logic_samples:
    logic_results = run_batch_pipeline(logic_samples, dataset_name="Logic Dataset")
    finalize_and_save(logic_results, OUTPUT_PATH, DATASET_PATH, "Logic Dataset")

# --- Physics ---
if PHYSICS_DATASET_PATH:
    physics_samples = load_dataset(PHYSICS_DATASET_PATH, is_physics=True)
    print(f"\nPhysics Dataset: {len(physics_samples)} samples")
    if physics_samples:
        po = OUTPUT_PATH.replace(".json", "_physics.json")
        physics_results = run_batch_pipeline(physics_samples, dataset_name="Physics Dataset")
        finalize_and_save(physics_results, po, PHYSICS_DATASET_PATH, "Physics Dataset")

print("\n" + "=" * 65)
print("  PIPELINE V3 HOAN TAT")
print("=" * 65)

''')

# ================================================================
# Clear all outputs, write
# ================================================================
for c in cells:
    c['outputs'] = []
    c['execution_count'] = None

nb['cells'] = cells

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nWrote {OUT}")
print(f"  Cells: {len(cells)}")
fsize = len(json.dumps(nb, ensure_ascii=False))
print(f"  Size: {fsize / 1024:.1f} KB")
print("Done!")
