import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('notebooka4ea9824c7.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']

# Print cell 6 in full (Stage 3 - Z3 Compiler)
src = cells[6]['source']
print(f"CELL 6 length: {len(src)} chars")
print("="*80)
print(src)
