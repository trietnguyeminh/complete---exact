"""
build_v5_2_final.py — Pipeline v5.2: Best-of-N + SpaCy Entity Extraction
_make_pass1_user is in STAGE 4 cell, not STAGE 2.
"""
import json

INPUT_NB  = r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v5_1_bon.ipynb"
OUTPUT_NB = r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v5_2_bon_spacy.ipynb"

with open(INPUT_NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes_made = []

for ci, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']

    # ── 1. Update header ───────────────────────────────────────────
    if 'vllm_pipeline_v5_1.py' in src:
        cell['source'] = src.replace(
            'vllm_pipeline_v5_1.py -- Neuro-Symbolic Pipeline (Best-of-N + Z3 Reward)',
            'vllm_pipeline_v5_2.py -- Neuro-Symbolic Pipeline (Best-of-N + SpaCy)'
        ).replace(
            'Pipeline v5.1 -- Best-of-N + Z3 Reward:',
            'Pipeline v5.2 -- Best-of-N + SpaCy Entity Extraction:'
        ).replace(
            'Key improvements over v4:',
            'Key improvements over v5.1:'
        ).replace(
            '  - Best-of-N: Generate N candidate formalizations (temp=0.6), Z3 picks best.\\n  - Eliminates sequential retry loops -- parallel sampling instead.\\n  - Dramatically reduces no_ast rate by giving the model N chances.',
            '  - Best-of-N: Generate N candidate formalizations (temp=0.6), Z3 picks best.\\n  - SpaCy NER: Extract entities from NL premises, inject into prompt.\\n  - Reduces hallucination by providing entity vocabulary to LLM.'
        )
        changes_made.append("header")

    # ── 2. Install cell: add spacy ─────────────────────────────────
    if 'STAGE 0 -- Install Dependencies & Config' in src:
        src = src.replace(
            '    "z3-solver",\n',
            '    "z3-solver",\n    "spacy",\n'
        )
        spacy_init = (
            '\n# --- SpaCy Setup ---\n'
            'import spacy\n'
            'try:\n'
            '    nlp = spacy.load("en_core_web_sm")\n'
            'except OSError:\n'
            '    spacy.cli.download("en_core_web_sm")\n'
            '    nlp = spacy.load("en_core_web_sm")\n'
            'print(f"SpaCy model: {nlp.meta[\'name\']} v{nlp.meta[\'version\']}")\n'
            'print("SpaCy OK")\n'
        )
        src = src.replace('print("Imports OK")\n', 'print("Imports OK")\n' + spacy_init)
        cell['source'] = src
        changes_made.append("spacy_install")

    # ── 3. Config: update output path ──────────────────────────────
    if 'CAU HINH -- Chinh sua o day' in src:
        src = src.replace(
            'OUTPUT_PATH    = "/kaggle/working/pipeline_results_v5_1.json"',
            'OUTPUT_PATH    = "/kaggle/working/pipeline_results_v5_2.json"'
        )
        cell['source'] = src
        changes_made.append("config")

    # ── 4. Stage 4: Add SpaCy + replace _make_pass1_user ──────────
    #    _make_pass1_user is in STAGE 4 cell (not STAGE 2!)
    if 'def _make_pass1_user(sample):' in src and 'STAGE 4' in src:
        func_start = src.find('def _make_pass1_user(sample):')
        func_end = src.find('\ndef _make_pass1_correction', func_start)

        if func_start >= 0 and func_end >= 0:
            spacy_and_new_func = (
                '# ------------------------------------------------------------------\n'
                '# SpaCy Entity Extraction (Direction D)\n'
                '# ------------------------------------------------------------------\n'
                'def extract_entities_spacy(premises_nl):\n'
                '    """Extract entities from NL premises using SpaCy."""\n'
                '    subjects, actions, objects = set(), set(), set()\n'
                '    for premise in premises_nl:\n'
                '        doc = nlp(premise)\n'
                '        for ent in doc.ents:\n'
                '            subjects.add(ent.text)\n'
                '        for token in doc:\n'
                '            if token.dep_ in ("nsubj", "nsubjpass") and token.pos_ in ("NOUN", "PROPN"):\n'
                '                subjects.add(token.text)\n'
                '            if token.pos_ == "VERB" and token.dep_ in ("ROOT", "relcl", "advcl", "xcomp", "ccomp"):\n'
                '                actions.add(token.lemma_.capitalize())\n'
                '            if token.dep_ in ("dobj", "pobj", "attr") and token.pos_ in ("NOUN", "PROPN"):\n'
                '                objects.add(token.text)\n'
                '            if token.pos_ == "ADJ" and token.dep_ in ("acomp", "attr"):\n'
                '                actions.add("Is" + token.text.capitalize())\n'
                '    return {"subjects": sorted(subjects), "actions": sorted(actions), "objects": sorted(objects)}\n'
                '\n\n'
                'def _make_pass1_user(sample):\n'
                '    premises = sample["premises-NL"]\n'
                '    numbered_p = "\\n".join(f"Premise {i+1}: {p}" for i, p in enumerate(premises))\n'
                '    entities = extract_entities_spacy(premises)\n'
                '    entity_hint = (\n'
                '        "=== ENTITY HINTS (tu SpaCy -- HAY dung lam ten Predicate) ===\\n"\n'
                '        f"  Subjects  : {\', \'.join(entities[\'subjects\']) or \'N/A\'}\\n"\n'
                '        f"  Actions   : {\', \'.join(entities[\'actions\']) or \'N/A\'}\\n"\n'
                '        f"  Objects   : {\', \'.join(entities[\'objects\']) or \'N/A\'}\\n"\n'
                '    )\n'
                '    return (\n'
                '        f"Hay hinh thuc hoa cac tien de sau.\\n\\n"\n'
                '        f"{entity_hint}\\n"\n'
                '        f"=== PREMISES ===\\n{numbered_p}"\n'
                '    )\n'
            )

            src = src[:func_start] + spacy_and_new_func + src[func_end:]
            changes_made.append("spacy_extraction + pass1_user REPLACED")
        else:
            changes_made.append(f"WARN: func_start={func_start}, func_end={func_end}")

        # Update labels
        src = src.replace(
            'Pipeline V5.1 (Best-of-N) Orchestrator loaded',
            'Pipeline V5.2 (Best-of-N + SpaCy) Orchestrator loaded'
        )
        src = src.replace(
            'PASS 1: FORMALIZE PREMISES (Best-of-{BEST_OF_N})',
            'PASS 1: FORMALIZE PREMISES (BoN={BEST_OF_N} + SpaCy)'
        )
        cell['source'] = src

    # ── 5. Stage 5: Update version labels ──────────────────────────
    if 'STAGE 5 -- Evaluation & Export' in src:
        src = src.replace(
            'EVALUATION SUMMARY (v5.1 BoN)',
            'EVALUATION SUMMARY (v5.2 BoN+SpaCy)'
        ).replace(
            '"pipeline_version": "v5.1_best_of_n"',
            '"pipeline_version": "v5.2_bon_spacy"'
        ).replace(
            'NEURO-SYMBOLIC PIPELINE v5.1 -- Best-of-N + Z3',
            'NEURO-SYMBOLIC PIPELINE v5.2 -- Best-of-N + SpaCy'
        ).replace(
            'PIPELINE V5.1 HOAN TAT',
            'PIPELINE V5.2 HOAN TAT'
        ).replace(
            'pipeline_results_v5_1.json',
            'pipeline_results_v5_2.json'
        ).replace(
            'pipeline_results_v5_1_physics.json',
            'pipeline_results_v5_2_physics.json'
        )
        cell['source'] = src
        changes_made.append("stage5_labels")

with open(OUTPUT_NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Changes: {changes_made}")
print(f"SUCCESS: Created {OUTPUT_NB}")
