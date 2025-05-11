import pandas as pd
import sys
import os
from pathlib import Path

def sample_qa_pairs(file_path, num_records):
    # Read the TSV file
    df = pd.read_csv(file_path, sep='\t')
    
    # Get the header
    header = df.columns.tolist()
    
    # Sample n records
    sampled_df = df.sample(n=num_records, random_state=42)
    
    # Create output filename
    input_path = Path(file_path)
    output_path = input_path.parent / f"{input_path.stem}_sampled_{num_records}{input_path.suffix}"
    
    # Write to new file
    sampled_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Created sampled file: {output_path}")
    print(f"Sampled {num_records} records from {len(df)} total records")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <file_path> <num_records>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    # Get the project root directory (assuming script is in project_folder/scripts)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Base directory for evaluation results
    file_path = os.path.join(project_root, file_path)
    
    try:
        num_records = int(sys.argv[2])
    except ValueError:
        print("Number of records must be an integer")
        sys.exit(1)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    sample_qa_pairs(file_path, num_records)