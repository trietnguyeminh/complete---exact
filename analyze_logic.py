import json, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')

# Load the dataset that the pipeline uses
path = r'C:\Users\Minh Triet\Downloads\Logic_Based_Educational_Queries (2).json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
print(f"Keys in sample[0]: {list(data[0].keys())}")

# Show 5 full samples
for i in [0, 1, 50, 100, 200]:
    if i >= len(data): break
    s = data[i]
    print(f"\n{'='*70}")
    print(f"SAMPLE {i}")
    print(f"{'='*70}")
    print(f"premises-NL ({len(s.get('premises-NL',[]))}):")
    for j, p in enumerate(s.get('premises-NL', [])):
        print(f"  P{j+1}: {p}")
    print(f"questions ({len(s.get('questions',[]))}):")
    for j, q in enumerate(s.get('questions', [])):
        print(f"  Q{j+1}: {q}")
    print(f"answers: {s.get('answers', [])}")
    # Check for other fields
    for k in s.keys():
        if k not in ('premises-NL', 'questions', 'answers'):
            val = str(s[k])[:100]
            print(f"  {k}: {val}")

# Analysis
print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")

# Question types
all_answers = []
for s in data:
    all_answers.extend(s.get('answers', []))

answer_types = Counter()
for a in all_answers:
    a_str = str(a).strip().upper()
    if a_str in ('A', 'B', 'C', 'D'):
        answer_types['MC (A/B/C/D)'] += 1
    elif a_str in ('YES', 'NO', 'UNKNOWN'):
        answer_types[f'Yes/No ({a_str})'] += 1
    elif a_str in ('TRUE', 'FALSE'):
        answer_types[f'True/False ({a_str})'] += 1
    else:
        answer_types[f'Other: {a_str[:30]}'] += 1

print(f"\nAnswer distribution ({len(all_answers)} total):")
for k, v in answer_types.most_common(20):
    print(f"  {k}: {v}")

# Premises per sample
premise_counts = [len(s.get('premises-NL', [])) for s in data]
print(f"\nPremises per sample: min={min(premise_counts)}, max={max(premise_counts)}, avg={sum(premise_counts)/len(premise_counts):.1f}")

# Questions per sample
q_counts = [len(s.get('questions', [])) for s in data]
print(f"Questions per sample: min={min(q_counts)}, max={max(q_counts)}, avg={sum(q_counts)/len(q_counts):.1f}")

# Analyze question patterns for entailment
entailment_patterns = Counter()
for s in data:
    for q in s.get('questions', []):
        q_low = q.lower()
        if 'true or false' in q_low or 'is it true' in q_low:
            entailment_patterns['true/false'] += 1
        elif any(w in q_low for w in ['can we conclude', 'does it follow', 'is it valid', 'can we infer']):
            entailment_patterns['entailment_check'] += 1
        elif any(w in q_low for w in ['which of the following', 'which one']):
            entailment_patterns['which_of'] += 1
        elif q_low.startswith(('is ', 'are ', 'does ', 'do ', 'can ', 'will ')):
            entailment_patterns['yes/no_question'] += 1
        elif any(w in q_low for w in ['what ', 'who ', 'where ', 'how ']):
            entailment_patterns['wh_question'] += 1
        else:
            entailment_patterns['other'] += 1

print(f"\nQuestion patterns:")
for k, v in entailment_patterns.most_common():
    print(f"  {k}: {v}")

# Show some Yes/No questions and their answers
print(f"\n{'='*70}")
print("SAMPLE YES/NO QUESTIONS (for Z3 entailment feasibility)")
print(f"{'='*70}")
count = 0
for s in data:
    for qi, q in enumerate(s.get('questions', [])):
        ans = s.get('answers', [])[qi] if qi < len(s.get('answers', [])) else '?'
        if str(ans).upper() in ('YES', 'NO', 'UNKNOWN', 'TRUE', 'FALSE'):
            print(f"\nPremises: {s['premises-NL']}")
            print(f"Q: {q}")
            print(f"A: {ans}")
            count += 1
            if count >= 8:
                break
    if count >= 8:
        break

# Show some MC questions
print(f"\n{'='*70}")
print("SAMPLE MC QUESTIONS")
print(f"{'='*70}")
count = 0
for s in data:
    for qi, q in enumerate(s.get('questions', [])):
        ans = s.get('answers', [])[qi] if qi < len(s.get('answers', [])) else '?'
        if str(ans).upper() in ('A', 'B', 'C', 'D'):
            print(f"\nPremises: {s['premises-NL'][:2]}...")
            print(f"Q: {q[:200]}")
            print(f"A: {ans}")
            count += 1
            if count >= 5:
                break
    if count >= 5:
        break
