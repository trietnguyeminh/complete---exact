import csv, json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'C:\Users\Minh Triet\Downloads\Physics_Problems.csv'
with open(path, 'r', encoding='utf-8') as f:
    raw = list(csv.DictReader(f))

# Analyze answer matching -- the comparison logic in the pipeline
# The pipeline does: str(ar["answer"]).strip().upper() == str(gt[...]).strip().upper()

# Even if model outputs "45", gt is also "45" -- should match?
# The problem is model is told to output A/B/C/D/Yes/No, so it never outputs "45"

# Check what kinds of answers exist
from collections import Counter

# Check if any are multiple choice format
has_options = 0
for r in raw:
    q = r.get('question', '')
    if any(opt in q for opt in ['A.', 'B.', 'C.', 'D.', '(A)', '(B)']):
        has_options += 1

print(f"Questions with MC options: {has_options}/{len(raw)}")

# Check answer diversity
ans_types = Counter()
for r in raw:
    ans = r.get('answer', '')
    if ans.replace('.','',1).replace('-','',1).isdigit():
        ans_types['simple_number'] += 1
    elif '×' in ans or '^' in ans:
        ans_types['scientific_notation'] += 1
    elif '√' in ans:
        ans_types['sqrt_expr'] += 1
    elif any(c.isalpha() for c in ans) and not any(c.isdigit() for c in ans):
        ans_types['text_only'] += 1
    else:
        ans_types['mixed'] += 1

print(f"\nAnswer types: {dict(ans_types)}")

# Check units
unit_counts = Counter(r.get('unit','') for r in raw)
print(f"\nUnit distribution (top 20): {unit_counts.most_common(20)}")

# Check text-only answers
text_answers = [r['answer'] for r in raw if any(c.isalpha() for c in r['answer']) and not any(c.isdigit() for c in r['answer'])]
print(f"\nText-only answers ({len(text_answers)}): {text_answers[:20]}")
