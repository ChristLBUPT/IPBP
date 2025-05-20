import os
import argparse
import json
# from transformers import LlamaForCausalLM

parser = argparse.ArgumentParser()
# positional argument for file name
parser.add_argument("file_name", type=str)

args = parser.parse_args()

def extract_python_code_from_ipynb(file_path):
    # Open and load the JSON content from the .ipynb file
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook_data = json.load(f)
    
    # Initialize an empty list to hold all the python code
    python_code_cells = []
    
    # Iterate through each cell in the notebook
    for idx, cell in enumerate(notebook_data['cells']):
        # Check if the cell type is 'code'
        if cell['cell_type'] == 'code':
            # Extract the 'source' field which contains the code lines
            code = ''.join(cell['source'])  # Combine code lines into a single string
            python_code_cells.append(f"====Cell {idx: 02d}====\n\n" + code)
    
    # Join all the code cells together (with optional separation for readability)
    all_python_code = '\n\n'.join(python_code_cells)
    
    return all_python_code

    
all_python_code = extract_python_code_from_ipynb(args.file_name)

print(all_python_code)