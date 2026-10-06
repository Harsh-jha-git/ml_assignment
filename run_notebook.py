import json
import os
import sys
import traceback
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nb_path = 'notebooks/01_heart_attack_prediction_pipeline.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

exec_globals = {}
exec_count = 1

print(f"Executing notebook '{nb_path}' top to bottom...")

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_code = "".join(cell['source'])
        print(f"--- Executing Code Cell {i+1} ---")
        
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        cell_outputs = []
        try:
            exec(source_code, exec_globals)
            stdout_text = redirected_output.getvalue()
            if stdout_text:
                cell_outputs.append({
                    'name': 'stdout',
                    'output_type': 'stream',
                    'text': stdout_text.splitlines(True)
                })
            cell['execution_count'] = exec_count
            cell['outputs'] = cell_outputs
            sys.stdout = old_stdout
            print(f"Cell {i+1} executed successfully.")
        except Exception as e:
            sys.stdout = old_stdout
            err_msg = traceback.format_exc()
            print(f"ERROR in Cell {i+1}:\n{err_msg}")
            sys.exit(1)
        finally:
            sys.stdout = old_stdout
            plt.close('all')
            
        exec_count += 1

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("\nNotebook executed and updated in-place successfully!")
