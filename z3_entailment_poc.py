"""
Z3 Entailment Proof-of-Concept
Demonstrate how Z3 can derive answers from FOL premises WITHOUT Qwen.
"""
import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')

# Simulated Z3 (we don't have z3-solver on Windows, but show the logic)

print("="*70)
print("Z3 ENTAILMENT-BASED ANSWER DERIVATION -- PROOF OF CONCEPT")
print("="*70)

# Load dataset
path = r'C:\Users\Minh Triet\Downloads\Logic_Based_Educational_Queries (2).json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Analyze question structure for Z3 feasibility
print("\n## ANSWER DISTRIBUTION ##")
from collections import Counter
answers = [str(a).strip().upper() for s in data for a in s.get('answers',[])]
c = Counter(answers)
for k,v in c.most_common():
    pct = v/len(answers)*100
    print(f"  {k:>8}: {v:>4} ({pct:.1f}%)  {'#'*int(pct)}")

total_yes_no = c['YES'] + c['NO'] + c['UNKNOWN']
total_mc = c['A'] + c['B'] + c['C'] + c['D']
print(f"\n  Yes/No/Unknown: {total_yes_no} ({total_yes_no/len(answers)*100:.1f}%)")
print(f"  MC (A/B/C/D):   {total_mc} ({total_mc/len(answers)*100:.1f}%)")

# Analyze question structure
print("\n## QUESTION STRUCTURE ANALYSIS ##")

# For Yes/No questions: Can we formalize the statement and check entailment?
yes_no_q = []
mc_q = []
for s in data:
    for qi, q in enumerate(s.get('questions', [])):
        ans = str(s.get('answers',[])[qi]).strip().upper() if qi < len(s.get('answers',[])) else '?'
        if ans in ('YES', 'NO', 'UNKNOWN'):
            yes_no_q.append((s, qi, q, ans))
        elif ans in ('A', 'B', 'C', 'D'):
            mc_q.append((s, qi, q, ans))

# Analyze Yes/No question patterns
print(f"\nYes/No/Unknown questions: {len(yes_no_q)}")
yn_patterns = Counter()
for s, qi, q, ans in yes_no_q:
    q_low = q.lower()
    if 'statement true' in q_low or 'is the following' in q_low:
        yn_patterns['Statement truth check'] += 1
    elif 'can we conclude' in q_low or 'can we infer' in q_low:
        yn_patterns['Entailment check'] += 1  
    elif q_low.startswith(('is ', 'are ', 'does ', 'do ', 'can ', 'will ')):
        yn_patterns['Direct yes/no'] += 1
    else:
        yn_patterns['Other'] += 1

for k,v in yn_patterns.most_common():
    print(f"  {k}: {v}")

# Analyze MC question patterns
print(f"\nMultiple choice questions: {len(mc_q)}")
mc_patterns = Counter()
for s, qi, q, ans in mc_q:
    q_low = q.lower()
    if 'which of the following' in q_low or 'which statement' in q_low:
        mc_patterns['Which statement'] += 1
    elif 'can be inferred' in q_low:
        mc_patterns['Inference check'] += 1
    else:
        mc_patterns['Other'] += 1

for k,v in mc_patterns.most_common():
    print(f"  {k}: {v}")

print(f"\n{'='*70}")
print("## Z3 ENTAILMENT STRATEGY ##")
print(f"{'='*70}")

print("""
INSIGHT: 67.7% of questions are Yes/No/Unknown (549/812).
These are PERFECT for Z3 entailment checking!

For Yes/No questions, the Z3 approach is:
  1. Formalize premises P1...Pn as Z3 assertions
  2. Formalize the question's statement Q as a Z3 expression
  3. Check entailment:
     - YES:     premises |= Q    (prove: premises + NOT(Q) is UNSAT)
     - NO:      premises |= ~Q   (prove: premises + Q is UNSAT)  
     - UNKNOWN: neither entailed  (both are SAT)

For MC questions (32.3%, 263/812):
  1. Formalize premises P1...Pn
  2. Formalize each option A, B, C, D as Z3 expressions
  3. For each option, check entailment
  4. The entailed option = answer

CRITICAL CHANGE NEEDED IN PIPELINE:
  Currently: Qwen formalizes ONLY premises, not questions
  Needed:    Qwen must ALSO formalize the question/statement/options
             Then Z3 checks entailment (NOT just consistency)

Current pipeline:
  Qwen → FOL(premises) → Z3 sat check → Qwen answers (unreliable)
  
Proposed pipeline:
  Qwen → FOL(premises) + FOL(question) → Z3 ENTAILMENT → deterministic answer

The key is: Z3 entailment is GUARANTEED correct if formalization is correct.
No hallucination. No guessing. Pure logic.
""")

# Check: in current pipeline, how many of the 150 sat samples got correct answers?
# We don't have the output file, but we know:
# - 150 sat, 381/812 correct (46.9%)
# - If Z3 entailment was used instead of Qwen answer extraction,
#   accuracy for sat samples could be 80-95% (limited only by formalization quality)

print("## EXPECTED IMPACT ##")
print(f"""
Current: 
  - 150/411 samples get Z3 sat (36.5%)
  - 381/812 questions correct (46.9%) -- using Qwen for answers

With Z3 entailment:
  - For 150 sat samples: Z3 can derive answer deterministically
  - Expected accuracy on sat samples: 80-95% 
    (limited only by formalization quality, NOT LLM reasoning)
  - For non-sat samples: fall back to Qwen (same as now)
  
Key improvements to increase sat rate:
  1. Better formalization prompt (few-shot with worked examples)
  2. Pre-validation catches structural errors early
  3. Smarter correction feedback
  Target: 60-70% sat rate → massive accuracy improvement
""")

# Show exactly what the formalization needs to output now
print("## FORMALIZATION CHANGE ##")
print("""
CURRENT output format (premises only):
{
  "step1_local_ontology": [...],
  "step2_premises_ast": [...]
}

PROPOSED output format (premises + question):
{
  "step1_local_ontology": [...],
  "step2_premises_ast": [...],
  "step3_question_ast": {           ← NEW
    "question_type": "yes_no",      ← "yes_no" or "multiple_choice"
    "statement_ast": { <AST> },     ← For yes/no: the statement to check
    "options_ast": {                 ← For MC: each option formalized
      "A": { <AST> },
      "B": { <AST> },
      "C": { <AST> },
      "D": { <AST> }
    }
  }
}
""")
