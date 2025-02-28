import os
import numpy as np
import pandas as pd

def analyze_tensors(filename):
    """
    Loads a NumPy array from the given filename, converts it to a DataFrame,
    and calculates the min, mean, median, and max for all rows. 
    Prints the results with file name and row designations.
    """

    try:
        # Load the NumPy array
        tensor = np.load(filename)

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(tensor)

        for col in df:
            print(f"{len(df[col])} total samples")
            print(f"  Column {col}:")
            print(f"    Min: {df[col].min():.7f}")
            print(f"    Mean: {df[col].mean():.7f}")
            print(f"    Std. Dev: {df[col].std():.7f}")
            print(f"    Median: {df[col].median():.7f}")
            print(f"    Max: {df[col].max():.7f}")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    filenames = ["target_tensors.npy"]
    for filename in filenames:
        analyze_tensors(filename)
