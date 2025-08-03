#!/usr/bin/env python
"""
Alternative way to run the notebook - converts it to Python script and executes
"""
import json
import sys

def extract_code_from_notebook(notebook_path):
    """Extract Python code cells from Jupyter notebook"""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    code_cells = []
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            # Join the source lines
            code = ''.join(cell['source'])
            if code.strip():  # Only add non-empty cells
                code_cells.append(f"# Cell {i+1}\n{code}\n")
    
    return '\n'.join(code_cells)

if __name__ == "__main__":
    notebook_path = 'toxic_comment_moderation.ipynb'
    
    # Extract code
    code = extract_code_from_notebook(notebook_path)
    
    # Save to a Python file
    with open('notebook_as_script.py', 'w') as f:
        f.write(code)
    
    print("Notebook converted to notebook_as_script.py")
    print("\nTo run the notebook code, execute:")
    print("python notebook_as_script.py")
    print("\nNote: Make sure you have train.csv in the current directory before running!")