import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')

OUT = r'C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v2_physics.ipynb'
with open(OUT, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix _normalize_number: use unicode-aware regex for stripping units
cell7 = nb['cells'][7]['source']

old = r"    s = re.sub(r'\s*[A-Za-z/]+\s*$', '', s).strip()"
new = r"    s = re.sub(r'\s*[A-Za-zμΩ°/%%]+\s*$', '', s).strip()"

if old in cell7:
    cell7 = cell7.replace(old, new)
    nb['cells'][7]['source'] = cell7
    print("Fixed _normalize_number unicode unit stripping")
else:
    print("Pattern not found -- checking...")
    # Show context
    idx = cell7.find("_normalize_number")
    print(cell7[idx:idx+300])

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Saved!")
