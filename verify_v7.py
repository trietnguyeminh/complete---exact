import json

def verify(path, label):
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    
    checks = {
        "v7_label": False,
        "think_strip": False,
        "func_cache_param": False,
        "shared_cache": False,
        "no_global_cache_clear": True,  # Should NOT have _func_cache.clear()
        "robust_qid": False,
        "z3_select_3_returns": False,
        "diagnostic_logging": False,
        "exception_not_swallowed": False,
        "lora_support": False,
        "bon_present": False,
    }
    
    all_src = ""
    for cell in nb['cells']:
        if cell['cell_type'] != 'code': continue
        src = cell['source']
        all_src += src + "\n"
    
    checks["v7_label"] = "v7" in all_src
    checks["think_strip"] = "<think>" in all_src
    checks["func_cache_param"] = "def compile_ast(node, var_map, func_cache):" in all_src
    checks["shared_cache"] = "_compile_premises_to_solver_shared" in all_src
    checks["no_global_cache_clear"] = "_func_cache.clear()" not in all_src
    checks["robust_qid"] = "int(qid_raw)" in all_src
    checks["z3_select_3_returns"] = "best_func_cache" in all_src
    checks["diagnostic_logging"] = "z3_methods" in all_src
    checks["exception_not_swallowed"] = "Z3 entailment error" in all_src
    checks["lora_support"] = "enable_lora" in all_src
    checks["bon_present"] = "batch_generate_bon" in all_src
    
    print(f"\n=== {label} ===")
    all_ok = True
    for name, ok in checks.items():
        status = "OK" if ok else "FAIL"
        if not ok: all_ok = False
        print(f"  [{status}] {name}")
    
    cell_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    print(f"  Total code cells: {cell_count}")
    return all_ok

ok1 = verify(r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v7_z3_standalone.ipynb", "V7 Z3 STANDALONE")
ok2 = verify(r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v7_inference.ipynb", "V7 INFERENCE (LoRA)")

print(f"\nAll OK: {ok1 and ok2}")
