import pandas as pd
import sys
import os

def calculate_max_absolute_error(file1_path, file2_path):
    try:
        # Load the CSV files into DataFrames
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)

        # Ensure both DataFrames have the same shape and columns
        if df1.shape != df2.shape:
            print("Error: The files do not have the same dimensions (rows/columns).")
            return

        # Ensure we are only working with numeric data
        # This drops non-numeric columns to avoid errors during subtraction
        df1_numeric = df1.select_dtypes(include=['number'])
        df2_numeric = df2.select_dtypes(include=['number'])

        if df1_numeric.shape != df2_numeric.shape:
             print("Error: Mismatch in numeric columns between files.")
             return

        # Calculate the absolute difference (Absolute Error)
        absolute_error = (df1_numeric - df2_numeric).abs()

        # Find the maximum error per column
        max_error_per_column = absolute_error.max()

        print("Maximum Absolute Error per Column:")
        print(max_error_per_column)

    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Replace these paths with your actual CSV file paths
    # You can also use sys.argv to pass arguments from the command line
    file_a = 'OriginalDATA\\CASE1.csv' 
    file_b = 'CASE1_final.csv'
    
    # Create dummy files for demonstration if they don't exist
    if not os.path.exists(file_a):
        pd.DataFrame({'A': [10, 20, 30], 'B': [1.1, 2.2, 3.3]}).to_csv(file_a, index=False)
    if not os.path.exists(file_b):
        pd.DataFrame({'A': [12, 19, 35], 'B': [1.0, 2.5, 3.3]}).to_csv(file_b, index=False)

    calculate_max_absolute_error(file_a, file_b)