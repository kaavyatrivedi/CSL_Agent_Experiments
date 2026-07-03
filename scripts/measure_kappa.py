# Save as scripts/measure_kappa.py
import yaml, json
from sklearn.metrics import cohen_kappa_score
import numpy as np
 
def extract_comp_type(contract_path):
    'Read YAML and return the comp.type field as a string'
    try:
        with open(contract_path) as f:
            c = yaml.safe_load(f)
        return c.get('comp', {}).get('type', 'UNKNOWN')
    except:
        return 'MISSING'
 
# Collect comp.type annotations from all 3 researchers for all 25 agents
r1, r2, r3 = [], [], []
for i in range(1, 26):
    aid = f'A{i:02d}'
    r1.append(extract_comp_type(f'ground_truth/{aid}_contract_R1.yaml'))
    r2.append(extract_comp_type(f'ground_truth/{aid}_contract_R2.yaml'))
    r3.append(extract_comp_type(f'ground_truth/{aid}_contract_R3.yaml'))
 
# Pairwise kappa
k12 = cohen_kappa_score(r1, r2)
k13 = cohen_kappa_score(r1, r3)
k23 = cohen_kappa_score(r2, r3)
avg_kappa = np.mean([k12, k13, k23])
 
print(f'Kappa R1-R2: {k12:.3f}')
print(f'Kappa R1-R3: {k13:.3f}')
print(f'Kappa R2-R3: {k23:.3f}')
print(f'Average kappa: {avg_kappa:.3f}')
print(f'Threshold met (>=0.80): {avg_kappa >= 0.80}')
