
"""
Check columns in the features CSV file
"""
import pandas as pd
import sys

def check_columns(file_path):
    """Check if high threshold target variables exist in the dataset"""
    print(f"Reading file: {file_path}")
    
    # Read only header to get column names
    try:
        df = pd.read_csv(file_path, nrows=0)
        print(f"Total columns: {len(df.columns)}")
        
        # Check for high threshold target columns
        high_threshold_cols = [col for col in df.columns if "high_threshold" in col]
        print(f"High threshold columns found: {len(high_threshold_cols)}")
        
        if high_threshold_cols:
            print("High threshold columns:")
            for col in high_threshold_cols:
                print(f"  - {col}")
        else:
            print("No high threshold columns found in the dataset.")
            
        # Check for other target columns
        target_cols = [col for col in df.columns if col.startswith("target_")]
        print(f"Total target columns: {len(target_cols)}")
        
        if target_cols:
            print("Target column types:")
            target_types = {}
            for col in target_cols:
                for key in ["binary", "threshold", "price_change", "direction"]:
                    if key in col:
                        target_types[key] = target_types.get(key, 0) + 1
            
            for key, count in target_types.items():
                print(f"  - {key}: {count}")
                
        return True
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

if __name__ == "__main__":
    file_path = "data/processed/btcusd_5m_features.csv"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    check_columns(file_path)
