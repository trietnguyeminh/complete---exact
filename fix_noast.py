import json

INPUT_NB = r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v4_update.ipynb"
OUTPUT_NB = r"C:\Users\Minh Triet\Desktop\SKIBIDI\notebook_v4_update_fixed.ipynb"

with open(INPUT_NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'PREMISE_FORMALIZATION_SYSTEM =' in cell['source']:
        source = cell['source']
        
        # Add the skeleton back to PREMISE_FORMALIZATION_SYSTEM
        target_str = '"Output JSON THUAN TUY (khong markdown, khong text thua):\\n"\n)'
        new_str = (
            '"Output JSON THUAN TUY (khong markdown, khong text thua):\\n"\n'
            "    '{\\n'\n"
            "    '  \"step1_local_ontology\": [\\n'\n"
            "    '    {\"predicate\": \"Name\", \"arity\": N, \"description\": \"...\"}\\n'\n"
            "    '  ],\\n'\n"
            "    '  \"step2_premises_ast\": [\\n'\n"
            "    '    {\"premise_id\": 0, \"source_nl\": \"...\", \"ast\": { <AST> }}\\n'\n"
            "    '  ]\\n'\n"
            "    '}\\n'\n"
            ")"
        )
        source = source.replace(target_str, new_str)
        
        # Add skeleton to QUESTION_FORMALIZATION_SYSTEM
        target_str2 = '"Output JSON THUAN TUY:\\n"\n)'
        new_str2 = (
            '"Output JSON THUAN TUY:\\n"\n'
            "    '{\\n'\n"
            "    '  \"step3_question_fol\": [\\n'\n"
            "    '    {\\n'\n"
            "    '      \"question_id\": 0,\\n'\n"
            "    '      \"question_type\": \"yes_no\",\\n'\n"
            "    '      \"source_nl\": \"statement to check\",\\n'\n"
            "    '      \"statement_ast\": { <AST> }\\n'\n"
            "    '    }\\n'\n"
            "    '  ]\\n'\n"
            "    '}\\n'\n"
            ")"
        )
        source = source.replace(target_str2, new_str2)
        
        cell['source'] = source

with open(OUTPUT_NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Fixed notebook saved to notebook_v4_update_fixed.ipynb")
