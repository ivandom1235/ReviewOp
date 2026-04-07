import sys
from pathlib import Path
import pandas as pd

# Add code dir to sys.path
sys.path.append(str(Path("c:/Users/MONISH/Desktop/GitHub Repo/ReviewOp/dataset_builder/code").resolve()))

try:
    from io_utils import load_inputs
    from build_dataset import _get_row_domain
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

input_dir = Path("c:/Users/MONISH/Desktop/GitHub Repo/ReviewOp/dataset_builder/input")
df = load_inputs(input_dir)

print(f"Total rows loaded: {len(df)}")
if not df.empty:
    if 'domain' in df.columns:
        print("\nDomain counts (from 'domain' column):")
        print(df['domain'].value_counts())
    
    # Check domain inference logic
    inferred_domains = df.apply(lambda row: _get_row_domain(row.to_dict()), axis=1)
    print("\nInferred domain counts (via _get_row_domain):")
    print(inferred_domains.value_counts())
else:
    print("DataFrame is empty!")
