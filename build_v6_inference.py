"""
build_v6_inference.py — Tao notebook inference v6 (v5.1 + Qwen3-8B + LoRA)
Patches v5.1 notebook to use Qwen3-8B with LoRA adapter from fine-tuning.
"""
import json

INPUT_NB = r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v5_1_bon.ipynb"
OUTPUT_NB = r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v6_inference.ipynb"

with open(INPUT_NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes = []

for ci, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']

    # ── 1. Header ──────────────────────────────────────────────────
    if 'vllm_pipeline_v5_1.py' in src:
        cell['source'] = src.replace(
            'vllm_pipeline_v5_1.py -- Neuro-Symbolic Pipeline (Best-of-N + Z3 Reward)',
            'vllm_pipeline_v6.py -- Neuro-Symbolic Pipeline (Qwen3-8B + LoRA + BoN)'
        ).replace(
            'Pipeline v5.1 -- Best-of-N + Z3 Reward:',
            'Pipeline v6 -- Fine-Tuned Qwen3-8B + Best-of-N + Z3:'
        ).replace(
            'Key improvements over v4:',
            'Key improvements over v5.1:'
        ).replace(
            '  - Best-of-N: Generate N candidate formalizations (temp=0.6), Z3 picks best.\\n  - Eliminates sequential retry loops -- parallel sampling instead.\\n  - Dramatically reduces no_ast rate by giving the model N chances.',
            '  - Qwen3-8B: Upgraded base model (stronger reasoning).\\n  - LoRA Adapter: QLoRA fine-tuned on logic reasoning (CoT).\\n  - Best-of-N: Generate N candidates, Z3 picks best.\\n  - Knowledge Distillation: Model trained on dataset explanations.'
        )
        changes.append("header")

    # ── 2. Config: change model + add LoRA path ────────────────────
    if 'CAU HINH -- Chinh sua o day' in src:
        # Change model ID
        src = src.replace(
            'QWEN_MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"',
            'QWEN_MODEL_ID  = "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1"'
        )
        # Add LoRA config after model ID
        lora_config = (
            '\n'
            '# --- LoRA Adapter (from notebook_v6_finetune) ---\n'
            'LORA_ADAPTER_PATH = "/kaggle/working/qwen3_logic_lora"\n'
            'ENABLE_LORA = True\n'
            'MAX_LORA_RANK = 16\n'
        )
        src = src.replace(
            'DATASET_PATH   =',
            lora_config + 'DATASET_PATH   ='
        )
        # Update output path
        src = src.replace(
            'OUTPUT_PATH    = "/kaggle/working/pipeline_results_v5_1.json"',
            'OUTPUT_PATH    = "/kaggle/working/pipeline_results_v6.json"'
        )
        cell['source'] = src
        changes.append("config")

    # ── 3. vLLM Engine: add LoRA support ───────────────────────────
    if 'STAGE 1 -- Load vLLM Engine' in src:
        # Replace LLM() call to include LoRA params
        old_llm = (
            'llm = LLM(\n'
            '    model=QWEN_MODEL_ID,\n'
            '    tensor_parallel_size=TENSOR_PARALLEL,\n'
            '    dtype=DTYPE,\n'
            '    max_model_len=MAX_MODEL_LEN,\n'
            '    gpu_memory_utilization=GPU_MEMORY_UTIL,\n'
            '    trust_remote_code=True,\n'
            '    enforce_eager=False,       # Dung CUDA graph de tang toc\n'
            ')'
        )
        new_llm = (
            'llm = LLM(\n'
            '    model=QWEN_MODEL_ID,\n'
            '    tensor_parallel_size=TENSOR_PARALLEL,\n'
            '    dtype=DTYPE,\n'
            '    max_model_len=MAX_MODEL_LEN,\n'
            '    gpu_memory_utilization=GPU_MEMORY_UTIL,\n'
            '    trust_remote_code=True,\n'
            '    enforce_eager=False,\n'
            '    enable_lora=ENABLE_LORA,\n'
            '    max_lora_rank=MAX_LORA_RANK if ENABLE_LORA else 8,\n'
            ')\n'
            '\n'
            '# --- LoRA Request ---\n'
            'lora_request = None\n'
            'if ENABLE_LORA and os.path.exists(LORA_ADAPTER_PATH):\n'
            '    from vllm.lora.request import LoRARequest\n'
            '    lora_request = LoRARequest("logic_lora", 1, LORA_ADAPTER_PATH)\n'
            '    print(f"LoRA adapter loaded: {LORA_ADAPTER_PATH}")\n'
            'else:\n'
            '    print("No LoRA adapter found, running base model")'
        )
        src = src.replace(old_llm, new_llm)

        # Add os import at top if not present
        if 'import os' not in src.split('STAGE 1')[0]:
            src = 'import os\n' + src

        cell['source'] = src
        changes.append("vllm_lora")

    # ── 4. Stage 2: Update batch_generate to use LoRA ──────────────
    if 'def batch_generate(' in src and 'STAGE 2' in src:
        # Update batch_generate to pass lora_request
        old_generate = 'outputs = llm.generate(formatted, params)'
        new_generate = 'outputs = llm.generate(formatted, params, lora_request=lora_request)'
        src = src.replace(old_generate, new_generate)

        # Update batch_generate_bon similarly
        old_bon_gen = (
            '    outputs = llm.generate(formatted, params)\n'
            '    outputs_sorted = sorted(outputs, key=lambda x: int(x.request_id))\n'
            '\n'
            '    return [\n'
            '        [o.text.strip() for o in out.outputs]\n'
            '        for out in outputs_sorted\n'
            '    ]'
        )
        new_bon_gen = (
            '    outputs = llm.generate(formatted, params, lora_request=lora_request)\n'
            '    outputs_sorted = sorted(outputs, key=lambda x: int(x.request_id))\n'
            '\n'
            '    return [\n'
            '        [o.text.strip() for o in out.outputs]\n'
            '        for out in outputs_sorted\n'
            '    ]'
        )
        src = src.replace(old_bon_gen, new_bon_gen)
        cell['source'] = src
        changes.append("batch_generate_lora")

    # ── 5. Stage 4: Update labels ──────────────────────────────────
    if 'STAGE 4 -- Batch Pipeline' in src:
        src = src.replace(
            'Pipeline V5.1 (Best-of-N) Orchestrator loaded',
            'Pipeline V6 (Qwen3-8B + LoRA + BoN) Orchestrator loaded'
        )
        cell['source'] = src
        changes.append("stage4")

    # ── 6. Stage 5: Update version labels ──────────────────────────
    if 'STAGE 5 -- Evaluation & Export' in src:
        src = src.replace(
            'EVALUATION SUMMARY (v5.1 BoN)',
            'EVALUATION SUMMARY (v6 LoRA+BoN)'
        ).replace(
            '"pipeline_version": "v5.1_best_of_n"',
            '"pipeline_version": "v6_lora_bon"'
        ).replace(
            'NEURO-SYMBOLIC PIPELINE v5.1 -- Best-of-N + Z3',
            'NEURO-SYMBOLIC PIPELINE v6 -- Qwen3-8B + LoRA + BoN'
        ).replace(
            'PIPELINE V5.1 HOAN TAT',
            'PIPELINE V6 HOAN TAT'
        ).replace(
            'pipeline_results_v5_1.json',
            'pipeline_results_v6.json'
        ).replace(
            'pipeline_results_v5_1_physics.json',
            'pipeline_results_v6_physics.json'
        )
        cell['source'] = src
        changes.append("stage5")

with open(OUTPUT_NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Changes: {changes}")
print(f"SUCCESS: Created {OUTPUT_NB}")
