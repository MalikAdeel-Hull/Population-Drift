import os

# Create folder structure
os.makedirs('src/drift_detection', exist_ok=True)

# File contents (I'll give you these next)
files_to_create = {
    'src/drift_detection/data.py': 'DATA_PY_CONTENT',
    'src/drift_detection/preprocessing.py': 'PREPROCESSING_PY_CONTENT',
    'src/drift_detection/drift.py': 'DRIFT_PY_CONTENT',
    'src/drift_detection/algorithms.py': 'ALGORITHMS_PY_CONTENT',
    'src/drift_detection/evaluation.py': 'EVALUATION_PY_CONTENT',
    'src/drift_detection/utils.py': 'UTILS_PY_CONTENT',
    'src/drift_detection/__init__.py': 'INIT_PY_CONTENT',
}

for filepath, content in files_to_create.items():
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✅ Created {filepath}")

print("\n✅ All modules created successfully!")