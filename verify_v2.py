import json, sys
sys.stdout.reconfigure(encoding='utf-8')

OUT = r'C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v2_physics.ipynb'

with open(OUT, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Notebook: {len(cells)} cells, nbformat={nb['nbformat']}")
print()

for i, c in enumerate(cells):
    src = c['source']
    lines = src.count('\n')
    print(f"Cell {i}: {c['cell_type']}, {lines} lines, {len(src)} chars")
    # Show first 3 meaningful lines
    meaningful = [l for l in src.split('\n') if l.strip() and not l.strip().startswith('#')]
    for l in meaningful[:2]:
        print(f"  > {l[:90]}")
    print()

# Verify key changes
print("=" * 60)
print("VERIFICATION CHECKS")
print("=" * 60)

# Cell 3: Config
c3 = cells[3]['source']
checks = [
    ("GPU_MEMORY_UTIL = 0.85" in c3 or "GPU_MEMORY_UTIL  = 0.85" in c3, "GPU_MEMORY_UTIL = 0.85"),
    ("MAX_NEW_TOKENS = 4096" in c3 or "MAX_NEW_TOKENS  = 4096" in c3, "MAX_NEW_TOKENS = 4096"),
    ("PHYSICS_MODE" in c3, "PHYSICS_MODE config"),
    ("PHYSICS_TOLERANCE" in c3, "PHYSICS_TOLERANCE config"),
]

# Cell 5: Prompts
c5 = cells[5]['source']
checks += [
    ("PHYSICS_SOLVER_SYSTEM" in c5, "Physics solver prompt"),
    ("PHYSICS_MC_SYSTEM" in c5, "Physics MC prompt"),
    ("FORMALIZATION_SYSTEM" in c5, "Original formalization prompt preserved"),
    ("CORRECTION_SYSTEM" in c5, "Original correction prompt preserved"),
    ("ANSWER_SYSTEM" in c5, "Original answer prompt preserved"),
    ("_is_physics" in c5, "Physics flag in load_dataset"),
    ("_unit" in c5, "Unit preservation in load_dataset"),
    ("_cot" in c5, "CoT preservation in load_dataset"),
]

# Cell 7: Pipeline
c7 = cells[7]['source']
checks += [
    ("_run_physics_pipeline" in c7, "Physics pipeline function"),
    ("_physics_answer_match" in c7, "Physics answer matching"),
    ("_normalize_number" in c7, "Number normalization"),
    ("if is_physics" in c7, "Physics routing in orchestrator"),
    ("_make_formalization_user" in c7, "Original formalization helper preserved"),
    ("_make_correction_user" in c7, "Original correction helper preserved"),
    ("_process_formalization" in c7, "Original process_formalization preserved"),
    ("ROUND 1" in c7, "Logic Round 1 preserved"),
    ("Batch Correction" in c7, "Logic correction loop preserved"),
    ("Batch Answer Extraction" in c7, "Logic answer extraction preserved"),
]

# Cell 8: Evaluation
c8 = cells[8]['source']
checks += [
    ('"skipped"' in c8, "Z3 'skipped' status added"),
    ("evaluate(" in c8, "evaluate() function preserved"),
    ("finalize_and_save(" in c8, "finalize_and_save() preserved"),
    ("PHYSICS_MODE" in c8, "Physics mode in run section"),
]

passed = 0
for ok, name in checks:
    status = "✓" if ok else "✗ FAIL"
    print(f"  {status}  {name}")
    if ok:
        passed += 1

print(f"\n  {passed}/{len(checks)} checks passed")
