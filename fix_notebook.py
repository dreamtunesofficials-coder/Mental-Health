import os
import json

# Read the notebook file
with open('notebooks/model_evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and replace the code in cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Convert list to string if needed
        if isinstance(source, list):
            source = ''.join(source)
        
        # Check if this is the cell we want to modify
        if "with open('./results/key_takeaways.txt', 'w', encoding='utf-8') as f:" in source:
            # Add the missing print statement
            old_text = '''with open('./results/key_takeaways.txt', 'w', encoding='utf-8') as f:
        f.write(key_takeaways)
    '''
            
            new_text = '''with open('./results/key_takeaways.txt', 'w', encoding='utf-8') as f:
        f.write(key_takeaways)
    print("\\n✅ Key takeaways saved to: ./results/key_takeaways.txt")
    '''
            
            cell['source'] = source.replace(old_text, new_text)
            print("Missing print statement added successfully!")

# Write back the notebook
with open('notebooks/model_evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook fixed successfully!')
