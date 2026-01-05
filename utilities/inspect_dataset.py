import pandas as pd
import sys
from pathlib import Path

def inspect(folder):
    p = Path(folder)
    meta_path = p / "metadata.csv"
    if not meta_path.exists():
        print(f"No metadata.csv found in {folder}")
        return

    df = pd.read_csv(meta_path)
    print(f"Total Samples: {len(df)}")
    
    print("\n--- By Method ---")
    print(df['method'].value_counts(dropna=False))
    
    print("\n--- By Split ---")
    print(df['split'].value_counts(dropna=False))
    
    print("\n--- By Is_Synthetic ---")
    print(df['is_synthetic'].value_counts(dropna=False))
    
    print("\n--- Cross Tab: Method vs Split ---")
    print(pd.crosstab(df['method'], df['split']))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        print("Usage: python inspect_dataset.py <processed_folder>")
