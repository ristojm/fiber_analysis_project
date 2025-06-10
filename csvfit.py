import pandas as pd

# Load and fix the CSV
df = pd.read_csv('multiprocessing_crumbly_results/parallel_evaluation_run_20250610_095835/ml_features_dataset.csv')

# Rename the column
df = df.rename(columns={'label': 'true_label'})

# Add text labels for better compatibility
label_map = {0: 'not', 1: 'intermediate', 2: 'crumbly'}
df['true_label_name'] = df['true_label'].map(label_map)

# Save the fixed CSV
df.to_csv('multiprocessing_crumbly_results/parallel_evaluation_run_20250610_095835/ml_features_dataset_fixed.csv', index=False)
print("Fixed CSV saved!")