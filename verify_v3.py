import json, sys
sys.stdout.reconfigure(encoding='utf-8')

OUT = r'C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v3_entailment.ipynb'
with open(OUT, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Notebook: {len(cells)} cells, nbformat={nb['nbformat']}")

for i, c in enumerate(cells):
    src = c['source']
    lines = src.count('\n')
    print(f"Cell {i}: {lines:>3} lines, {len(src):>5} chars")

print("\n" + "="*60)
print("VERIFICATION CHECKS")
print("="*60)

c3 = cells[3]['source']
c5 = cells[5]['source']
c6 = cells[6]['source']
c7 = cells[7]['source']
c8 = cells[8]['source']

checks = [
    # Config
    ("Z3_ENTAILMENT" in c3, "Z3_ENTAILMENT config flag"),
    ("Z3_SOLVER_TIMEOUT" in c3, "Z3_SOLVER_TIMEOUT config"),
    ("GPU_MEMORY_UTIL  = 0.85" in c3, "GPU_MEMORY_UTIL = 0.85"),
    
    # Prompts
    ("step3_question_fol" in c5, "Step 3 in formalization prompt"),
    ("FEW_SHOT_EXAMPLE" in c5, "Few-shot example in prompt"),
    ("statement_ast" in c5, "statement_ast in prompt format"),
    ("question_type" in c5, "question_type in prompt format"),
    ("options_ast" in c5, "options_ast in MC format"),
    ("FORMALIZATION_SYSTEM" in c5, "Formalization system prompt"),
    ("CORRECTION_SYSTEM" in c5, "Correction system prompt"),
    ("ANSWER_SYSTEM" in c5, "Answer system prompt (fallback)"),
    ("PHYSICS_SOLVER_SYSTEM" in c5, "Physics solver prompt"),
    
    # Z3 Entailment (Cell 6)
    ("z3_entailment_check" in c6, "z3_entailment_check() function"),
    ("_entail_yes_no" in c6, "_entail_yes_no() function"),
    ("_entail_mc" in c6, "_entail_mc() function"),
    ("_compile_premises_to_solver" in c6, "_compile_premises_to_solver()"),
    ("Not(stmt_expr)" in c6, "NOT(statement) for entailment"),
    ("Not(opt_expr)" in c6, "NOT(option) for MC entailment"),
    ("compile_ast" in c6, "compile_ast() preserved"),
    ("verify_with_z3" in c6, "verify_with_z3() preserved"),
    ("hallucination_check" in c6, "hallucination_check() preserved"),
    
    # Pipeline (Cell 7)
    ("question_fol" in c7, "question_fol in PipelineResult"),
    ("answer_source" in c7, "answer_source tracking"),
    ("step3_question_fol" in c7, "step3 parsing in _process_formalization"),
    ("z3_entailment_check" in c7, "Z3 entailment called in pipeline"),
    ("qwen_fallback_items" in c7, "Qwen fallback routing"),
    ("Z3_ENTAILMENT" in c7, "Z3_ENTAILMENT flag check"),
    ("_run_physics_pipeline" in c7, "Physics pipeline preserved"),
    ("_make_formalization_user" in c7, "Formalization user prompt"),
    ("premises + questions" in c7.lower() or "premises va cau hoi" in c7.lower() 
     or "QUESTIONS" in c7, "Questions included in formalization"),
    
    # Evaluation (Cell 8)
    ("z3_entailment_accuracy" in c8, "Z3 entailment accuracy metric"),
    ("qwen_accuracy" in c8, "Qwen accuracy metric"),
    ("answer_source" in c8, "Answer source in output"),
    ("v3_entailment" in c8, "Version tag v3"),
    ('"skipped"' in c8, "Skipped status for physics"),
]

passed = 0
for ok, name in checks:
    s = "✓" if ok else "✗ FAIL"
    print(f"  {s}  {name}")
    if ok: passed += 1

print(f"\n  {passed}/{len(checks)} checks passed")
