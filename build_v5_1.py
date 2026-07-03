"""
build_v5_1.py — Pipeline v5.1: Best-of-N Sampling + Z3 Reward
(Direction B only, NO SpaCy)
"""
import json

INPUT_NB  = r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v4_update_fixed.ipynb"
OUTPUT_NB = r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v5_1_bon.ipynb"

with open(INPUT_NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']

    # ── 1. Update header cell ──────────────────────────────────────
    if 'vllm_pipeline_v4.py' in src:
        cell['source'] = src.replace(
            'vllm_pipeline_v4.py -- Neuro-Symbolic Pipeline (Two-Pass Formalization)',
            'vllm_pipeline_v5_1.py -- Neuro-Symbolic Pipeline (Best-of-N + Z3 Reward)'
        ).replace(
            'Pipeline v4 -- Two-Pass Formalization:',
            'Pipeline v5.1 -- Best-of-N + Z3 Reward:'
        ).replace(
            'Key improvements over v3:',
            'Key improvements over v4:'
        ).replace(
            '  - Splits formalization into 2 passes to avoid LLM overload & VRAM OOM crashes.\\n  - Pass 1: Formalize Premises (high SAT rate).\\n  - Pass 2: Formalize Questions (using proven Ontology).',
            '  - Best-of-N: Generate N candidate formalizations (temp=0.6), Z3 picks best.\\n  - Eliminates sequential retry loops -- parallel sampling instead.\\n  - Dramatically reduces no_ast rate by giving the model N chances.'
        )

    # ── 2. Config cell: add BoN params ─────────────────────────────
    if 'CAU HINH -- Chinh sua o day' in src:
        bon_config = (
            '\n# --- Best-of-N Config ---\n'
            'BEST_OF_N       = 5     # Number of candidates per sample\n'
            'BON_TEMPERATURE = 0.6   # Higher temp for diversity\n'
        )
        src = src.replace(
            "N_SAMPLES      = 411",
            "N_SAMPLES      = 411" + bon_config
        )
        # Update output path
        src = src.replace(
            'OUTPUT_PATH    = "/kaggle/working/pipeline_results_v4.json"',
            'OUTPUT_PATH    = "/kaggle/working/pipeline_results_v5_1.json"'
        )
        cell['source'] = src

    # ── 3. Stage 2: Add batch_generate_bon + z3_select_best ────────
    if 'STAGE 2 -- Ontology & Prompts' in src:
        # Add new functions after batch_generate
        bon_funcs = '''

def batch_generate_bon(prompt_pairs, max_tokens, n=BEST_OF_N):
    """Generate N candidates per prompt using higher temperature."""
    formatted = []
    for sys_msg, usr_msg in prompt_pairs:
        messages = [{"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg}]
        formatted.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    params = SamplingParams(
        temperature=BON_TEMPERATURE,
        max_tokens=max_tokens,
        repetition_penalty=1.1,
        n=n,
    )
    outputs = llm.generate(formatted, params)
    outputs_sorted = sorted(outputs, key=lambda x: int(x.request_id))

    return [
        [o.text.strip() for o in out.outputs]
        for out in outputs_sorted
    ]


def z3_select_best(candidates):
    """
    Try each candidate through Z3, pick the best one.
    Priority: sat > unsat > unknown > compile_error > no_ast
    """
    PRIORITY = {"sat": 4, "unsat": 3, "unknown": 2, "compile_error": 1, "no_ast": 0}
    best_data, best_status, best_score = {}, "no_ast", -1

    for raw in candidates:
        data = safe_json(raw)
        premises_ast = data.get("step2_premises_ast", [])

        if not premises_ast:
            score = PRIORITY["no_ast"]
            status = "no_ast"
        else:
            z3_info = verify_with_z3(premises_ast)
            status = z3_info["status"]
            score = PRIORITY.get(status, 0)

        if score > best_score:
            best_score = score
            best_status = status
            best_data = data

        if best_score >= 4:  # sat -> stop early
            break

    return best_data, best_status
'''
        src = src.rstrip() + bon_funcs + '\n'
        cell['source'] = src

    # ── 4. Stage 4: Replace Pass 1 with Best-of-N ─────────────────
    if 'STAGE 4 -- Batch Pipeline' in src:
        # Replace the old Pass 1 block
        old_pass1 = '''    print(f"\\n{'=' * 60}\\n  PASS 1: FORMALIZE PREMISES\\n{'=' * 60}")
    t1 = time.time()
    form_prompts = [(PREMISE_FORMALIZATION_SYSTEM, _make_pass1_user(s)) for s in samples]
    form_responses = batch_generate(form_prompts, MAX_NEW_TOKENS_PASS1)
    
    for i, raw in enumerate(form_responses):
        results[i].z3_attempts = 1
        try: _process_pass1(results[i], raw)
        except Exception as e: results[i].z3_status = "no_ast"

    # Correction Loop (Pass 1)
    for attempt in range(2, MAX_RETRIES + 1):
        failed = [i for i in range(N) if results[i].z3_status == "compile_error"]
        if not failed: break
        print(f"  [Round {attempt}] Correction -- {len(failed)} failed")
        corr_prompts = [(CORRECTION_SYSTEM, _make_pass1_correction_user(samples[i], results[i])) for i in failed]
        corr_responses = batch_generate(corr_prompts, MAX_NEW_TOKENS_PASS1)
        for j, raw in enumerate(corr_responses):
            idx = failed[j]
            results[idx].z3_attempts = attempt
            try: _process_pass1(results[idx], raw)
            except: pass

    status_counts = Counter(r.z3_status for r in results)
    print(f"  Pass 1 Z3 Status: {dict(status_counts)}")'''

        new_pass1 = '''    print(f"\\n{'=' * 60}\\n  PASS 1: FORMALIZE PREMISES (Best-of-{BEST_OF_N})\\n{'=' * 60}")
    t1 = time.time()
    form_prompts = [(PREMISE_FORMALIZATION_SYSTEM, _make_pass1_user(s)) for s in samples]
    all_candidates = batch_generate_bon(form_prompts, MAX_NEW_TOKENS_PASS1, n=BEST_OF_N)

    for i, candidates in enumerate(all_candidates):
        results[i].z3_attempts = 1
        try:
            best_data, best_status = z3_select_best(candidates)
            results[i].local_ontology = best_data.get("step1_local_ontology", [])
            results[i].premises_ast = best_data.get("step2_premises_ast", [])
            results[i].z3_status = best_status

            if results[i].premises_ast:
                hw = hallucination_check(results[i].local_ontology, results[i].premises_ast)
                results[i].hallucination_warn = hw
                z3_info = verify_with_z3(results[i].premises_ast)
                results[i].z3_errors = z3_info.get("errors", [])
                results[i].z3_compiled = z3_info.get("compiled_count", 0)
                results[i].z3_total = z3_info.get("total_count", 0)
        except Exception as e:
            results[i].z3_status = "no_ast"

    status_counts = Counter(r.z3_status for r in results)
    print(f"  Pass 1 Z3 Status (BoN={BEST_OF_N}): {dict(status_counts)}")'''

        src = src.replace(old_pass1, new_pass1)
        # Update pipeline loaded message
        src = src.replace('Pipeline V4 Orchestrator loaded', 'Pipeline V5.1 (Best-of-N) Orchestrator loaded')
        cell['source'] = src

    # ── 5. Stage 5: Update version labels ──────────────────────────
    if 'STAGE 5 -- Evaluation & Export' in src:
        src = src.replace(
            'EVALUATION SUMMARY (v3)',
            'EVALUATION SUMMARY (v5.1 BoN)'
        ).replace(
            '"pipeline_version": "v3_entailment"',
            '"pipeline_version": "v5.1_best_of_n"'
        ).replace(
            'NEURO-SYMBOLIC PIPELINE v3 -- Z3 ENTAILMENT',
            'NEURO-SYMBOLIC PIPELINE v5.1 -- Best-of-N + Z3'
        ).replace(
            'PIPELINE V3 HOAN TAT',
            'PIPELINE V5.1 HOAN TAT'
        ).replace(
            'pipeline_results_v4.json',
            'pipeline_results_v5_1.json'
        ).replace(
            'pipeline_results_v4_physics.json',
            'pipeline_results_v5_1_physics.json'
        )
        cell['source'] = src

with open(OUTPUT_NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"SUCCESS: Created {OUTPUT_NB}")
