# EXACT 2026 — Neuro-Symbolic Educational QA System

Team **Astatine**'s cleaned submission archive for the **EXACT 2026 / IJCNN 2026 Explainable AI Challenge**.

This repository consolidates the core work from five development-stage repositories into a single readable entry point.  
The system explores a **neuro-symbolic educational QA pipeline** that combines open-source LLM reasoning, Qwen/Qwen3 fine-tuning, Best-of-N inference, Z3-style symbolic verification, local smoke testing, risk auditing, and final submission hardening.

---

## Round 2 Snapshot

```text
Team: Astatine
Rank: #28 / 180 registered teams
Total: 34.27 pts
Penalty: 0 pts
Progress: 50 / 50
Type 1: 8.62 pts
Type 2: 24.00 pts
Time: 1.65 pts
Data: Pending
```

> The score above is recorded from a Round 2 real-time leaderboard snapshot.  
> It should be interpreted as a competition snapshot, not a final post-competition official ranking unless the final leaderboard confirms it.

---

## Project Summary

EXACT focuses on transparent educational question answering. The system must not only produce answers, but also support explainable reasoning over educational logic and STEM-style questions.

The project covers two major task types:

| Task Type | Focus | System Direction |
|---|---|---|
| **Type 1** | Logic-based educational regulation QA | LLM formalization + symbolic verification + answer parsing |
| **Type 2** | STEM / physics educational QA | Open-source LLM reasoning + prompt control + evaluation pipeline |

The main engineering idea is:

```text
LLM reasoning is flexible but can hallucinate.
Symbolic reasoning is reliable but depends on correct formalization.
Smoke tests and audits prevent silent regression before submission.
```

---

## System Pipeline

```text
Input dataset
   │
   ├── Natural-language premises
   ├── FOL / symbolic annotations when available
   ├── Multiple-choice questions
   └── Yes / No / Unknown questions
        │
        ▼
[1] Dataset Preflight Audit
        │
        ├── flatten grouped samples
        ├── inspect label distribution
        ├── detect MC ambiguity
        ├── check prompt-row identity
        └── flag risky dataset versions
        │
        ▼
[2] Prompt & Formalization Layer
        │
        ├── construct task-specific prompts
        ├── normalize premise/question format
        ├── formalize premises into logic-like structures
        ├── optionally formalize questions/options
        └── enforce strict answer schema
        │
        ▼
[3] LLM Reasoning Layer
        │
        ├── Qwen / Qwen3 open-source model inference
        ├── Qwen3-8B QLoRA logic fine-tuning
        ├── response-only supervised fine-tuning
        ├── chain-of-thought style training targets
        └── Best-of-N candidate generation
        │
        ▼
[4] Neuro-Symbolic Verification Layer
        │
        ├── parse generated formal structures
        ├── run Z3-style satisfiability / entailment checks
        ├── verify Yes / No / Unknown statements
        ├── evaluate MC options when formalized
        └── fall back safely when symbolic confidence is low
        │
        ▼
[5] Candidate Selection & Calibration
        │
        ├── compare LLM candidates
        ├── prefer verified symbolic outputs
        ├── apply strict parser policy
        ├── repair known formatting failures
        └── abstain or preserve safer output when uncertain
        │
        ▼
[6] Local Evaluation & Smoke Gate
        │
        ├── evaluate generated predictions
        ├── compute accuracy and F1 metrics
        ├── compare old vs new pipeline versions
        ├── enforce allowed-drop thresholds
        └── generate risk reports
        │
        ▼
[7] Submission Hardening
        │
        ├── run dataset preflight audit
        ├── run prompt identity audit
        ├── export final predictions
        ├── generate summary / manifest files
        └── package final deliverables
```

---

## Why Neuro-Symbolic?

The repository investigates a practical hybrid design:

```text
LLM = language understanding, formalization, candidate generation
Z3-style reasoning = deterministic verification when formalization is reliable
Best-of-N = candidate diversity
Parser policy = valid answer enforcement
Smoke gate = regression protection
Risk audit = submission safety
```

For Yes / No / Unknown questions, the intended symbolic strategy is:

```text
YES      if premises entail Q
NO       if premises entail not-Q
UNKNOWN  if neither Q nor not-Q is entailed
```

For multiple-choice questions:

```text
Formalize options A, B, C, D
Check which option is entailed by the premises
Return the verified option when confidence is high
Otherwise fall back to the safer LLM/parser output
```

---

## Consolidated Source Repositories

This repository is the cleaned final entry point for work that originally evolved across five repositories:

1. <https://github.com/trietnguyeminh/XAI-IJCNN>
2. <https://github.com/trietnguyeminh/EXACT-IJCNN-2026>
3. <https://github.com/trietnguyeminh/XAI-IJCNN-finish>
4. <https://github.com/trietnguyeminh/Submission---EXACT--IJCNN>
5. <https://github.com/trietnguyeminh/Submission---EXACT--IJCNN---v2>

Recommended reading order:

```text
XAI-IJCNN
   ↓
EXACT-IJCNN-2026
   ↓
XAI-IJCNN-finish
   ↓
Submission---EXACT--IJCNN
   ↓
Submission---EXACT--IJCNN---v2
   ↓
complete---exact
```

---

## Repository Structure

The repository keeps a curated set of core files rather than the entire noisy development history.

```text
.
├── README.md
├── DATASET_DATA_TYPE_1/
├── EXACT/
├── archive/
├── files/
├── results (5)/
├── source_code/
├── build_v*.py
├── verify_v*.py
├── notebook_v*.ipynb
├── exact_70*.ipynb
├── generated_*.json
├── 06*_summary.json
├── 06*_risk_report.json
├── 06*_smoke_gate.json
├── MANIFEST.json
└── SUMMARY.json
```

---

## Key Components

### 1. Dataset Audit

Relevant files:

```text
06_dataset_preflight_audit.json
06_prompt_current_row_identity_audit.json
Logic_Based_Educational_Queries_final.json
Logic_Based_Educational_Queries_v5_repair_tier1_dedup_normfol_drop_logicfallacy.json
```

This stage checks dataset shape, flattened sample counts, multiple-choice ambiguity, benchmark version risk, and prompt-row identity consistency.

A documented audit detected ambiguity in multiple-choice variants, which helped prevent evaluation mistakes caused by using an outdated or risky benchmark version.

---

### 2. Qwen3 QLoRA Fine-Tuning

Relevant files:

```text
build_v6_finetune.py
notebook_v6_finetune.ipynb
notebook_v14_cot_lora_finetune.ipynb
```

The fine-tuning stage builds a Kaggle-ready notebook for Qwen3-8B logic reasoning.

Main ideas:

- Qwen3-8B base model
- Unsloth QLoRA
- 4-bit model loading
- LoRA rank 16
- response-only supervised fine-tuning
- chain-of-thought style training targets
- oversampling of `Unknown` answers
- train / validation / reserved test split

---

### 3. vLLM + LoRA Inference

Relevant files:

```text
build_v6_inference.py
notebook_v6_inference.ipynb
notebook_v7_inference.ipynb
notebook_v5_1_bon.ipynb
```

The inference stage upgrades the earlier Best-of-N pipeline into a Qwen3-8B + LoRA pipeline.

Main ideas:

- vLLM batch inference
- LoRA adapter loading
- Best-of-N candidate generation
- prompt-controlled structured answers
- versioned output files
- evaluation/export stage

---

### 4. Z3 / Symbolic Verification

Relevant files:

```text
z3_entailment_poc.py
exact_70_plus_z3.ipynb
exact_70_z3_hybrid.ipynb
notebook_v7_z3_standalone.ipynb
02_symbolic_analysis_mc_and_statement.ipynb
```

This stage explores deterministic entailment-based answer derivation.

The key insight is that Z3-style checking can produce reliable answers when the premises, questions, and options are formalized correctly. The symbolic layer is intentionally conservative: when formalization quality is uncertain, the system avoids unsafe overrides.

---

### 5. Candidate Selection and Parser Policy

Relevant files:

```text
03_master_selector_from_analysis.ipynb
03_generated_v4style_300_master_selector_summary.json
03_master_risk_report.json
fix_noast.py
test_matching.py
verify_v*.py
```

This layer selects among generated candidates, repairs known formatting issues, enforces valid labels, and reduces invalid-answer risk.

---

### 6. Smoke Gate and Regression Testing

Relevant files:

```text
06b_combined_summary.json
06b_quick_evaluated_output_report.json
06b_quick_local_simulation_report.json
06b_risk_report.json
06b_smoke_gate.json
```

Recorded smoke-gate result:

```text
Dataset: generated_v4style_300
Smoke size: 50
Old correct: 41 / 50
New correct: 44 / 50
Accuracy: 0.88
Micro F1: 0.88
Macro-7 F1: 0.7809
Weighted F1: 0.8734
Prompt identity match: 100%
Gate: pass
```

This stage is important because it protects the submission from silent regressions when new logic or parser changes are introduced.

---

### 7. Final Safe-Audit Stage

Relevant files:

```text
MANIFEST.json
SUMMARY.json
stage_00_current_pipeline.json
stage_00_input_audit.json
stage_01_replay_report_selected.json
stage_02_full_metrics.json
stage_03_gap_audit.json
stage_04_overfit_underfit_audit.json
stage_05_upgrade_stage_reports.json
stage_06_safe_upgrade_decision.json
```

Recorded final audit version:

```text
v40.5-one-run-safe-audit
```

Replay summary:

```text
Replay cases: 25
Fired: 5
Correct when fired: 5
Precision when fired: 1.0
Coverage: 0.2
Gate: ABSTAIN_SAFE
```

This means the safe-audit layer preferred low coverage with high precision rather than aggressive corrections that could introduce wrong answers.

---

## Development Timeline

| Stage | Repository | Role |
|---|---|---|
| Early prototyping | `XAI-IJCNN` | Physics notebook, entailment notebook, Best-of-N, early Z3 experiments |
| Script reconstruction | `EXACT-IJCNN-2026` | Build scripts, verification scripts, Qwen3 fine-tune/inference generation |
| Hybrid development | `XAI-IJCNN-finish` | Dataset variants, `exact_70`, Z3 hybrid notebooks, formalizer/fine-tuning experiments |
| Main workbench | `Submission---EXACT--IJCNN` | Large submission workbench, selector logic, risk reports, full-test artifacts |
| Final packaging | `Submission---EXACT--IJCNN---v2` | Cleanup, smoke gates, audit reports, final deliverables |
| Curated archive | `complete---exact` | Cleaned readable index for portfolio and review |

---

## How to Read This Repository

If you only have a few minutes, read in this order:

1. `README.md` — project story and pipeline
2. `06b_combined_summary.json` — smoke-gate metrics
3. `SUMMARY.json` — final safe-audit summary
4. `build_v6_finetune.py` — Qwen3 QLoRA fine-tuning notebook builder
5. `build_v6_inference.py` — vLLM + LoRA inference pipeline patch
6. `z3_entailment_poc.py` — symbolic reasoning proof-of-concept

If you want the development history, inspect the notebook families:

```text
notebook_v2_physics.ipynb
notebook_v3_entailment.ipynb
notebook_v5_1_bon.ipynb
notebook_v6_finetune.ipynb
notebook_v6_inference.ipynb
notebook_v7_z3_standalone.ipynb
exact_70_plus_z3.ipynb
exact_70_z3_hybrid.ipynb
```

---

## Results Summary

### Competition Snapshot

```text
Team Astatine
Round 2 rank: #28 / 180 registered teams
Total score: 34.27
Penalty: 0
Progress: 50 / 50
```

### Local Smoke Test

```text
Accuracy: 0.88
Micro F1: 0.88
Macro-7 F1: 0.7809
Weighted F1: 0.8734
Prompt identity match: 100%
Gate: pass
```

### Safe-Audit Replay

```text
Version: v40.5-one-run-safe-audit
Fired: 5 / 25
Correct when fired: 5 / 5
Precision when fired: 1.0
Coverage: 0.2
Gate: ABSTAIN_SAFE
```

---

## Strengths

- End-to-end AI reasoning system, not just a single notebook.
- Uses open-source LLMs instead of closed commercial APIs.
- Includes Qwen3-8B QLoRA fine-tuning.
- Includes vLLM inference and LoRA adapter integration.
- Explores neuro-symbolic verification through Z3-style entailment.
- Includes prompt identity checks, smoke gates, risk reports, and safe-audit logic.
- Preserves development history while offering a curated entry point.
- Shows real competition iteration under time pressure.

---

## Limitations

- The repository is still notebook-heavy.
- Some scripts generate notebooks instead of running as reusable modules.
- Some paths are Kaggle- or Windows-specific.
- The symbolic layer is conservative and has limited coverage.
- There is not yet a one-command reproduction script.
- Raw competition data and runtime environment may not be fully included.

---

## Recommended Next Refactor

A cleaner production-style structure would be:

```text
complete---exact/
├── README.md
├── docs/
│   ├── timeline.md
│   ├── system_pipeline.md
│   └── audit_reports.md
├── src/
│   ├── data/
│   ├── prompts/
│   ├── inference/
│   ├── symbolic/
│   ├── evaluation/
│   └── submission/
├── notebooks/
│   ├── prototypes/
│   ├── finetune/
│   └── inference/
├── artifacts/
│   ├── summaries/
│   ├── risk_reports/
│   └── smoke_tests/
├── requirements.txt
└── run_smoke_test.py
```

Minimum next steps:

1. Add `requirements.txt`.
2. Add `run_smoke_test.py`.
3. Move historical notebooks into `notebooks/`.
4. Move JSON reports into `artifacts/`.
5. Add a pipeline diagram image.
6. Add `docs/timeline.md`.
7. Add an inference-only script for a small sample.

---

## Portfolio Summary

```text
Built a neuro-symbolic educational QA system for EXACT 2026 / IJCNN, combining Qwen3-8B QLoRA fine-tuning, vLLM inference, Best-of-N generation, Z3-style symbolic verification, strict answer parsing, smoke testing, risk auditing, and final submission hardening. Reached Round 2 snapshot rank #28 / 180 registered teams as Team Astatine with 34.27 points and 0 penalty.
```

---

## Disclaimer

This repository documents a research and competition system for explainable educational question answering.  
It is not intended as a production tutoring system without further validation.
