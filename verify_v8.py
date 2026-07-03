import json

with open(r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v8.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

all_src = " ".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")

checks = {
    "v8_label": "v8" in all_src,
    "const_map": "_get_constant_val" in all_src,
    "fuzzy_match": "[FUZZY]" in all_src,
    "self_correction": "SELF_CORRECTION_SYSTEM" in all_src,
    "sc_pipeline": "SC OK" in all_src,
    "elimination": "z3_elimination" in all_src,
    "no_hash": "hash(name)" not in all_src,
    "shared_cache": "_compile_premises_to_solver_shared" in all_src,
    "const_in_select": "best_const_map" in all_src,
    "z3_sc_count": "z3_sc_answers" in all_src,
}

for k, v in checks.items():
    status = "OK" if v else "FAIL"
    print(f"  [{status}] {k}")

cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
print(f"\nCells: {cells}")
print(f"All OK: {all(checks.values())}")
