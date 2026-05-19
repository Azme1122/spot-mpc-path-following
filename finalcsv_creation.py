import pandas as pd

def merge_specific_columns(source_file_1, source_file_2, output_file):
    """
    Reads two CSV files, takes specific columns from the second file,
    updates the first file's data with them, and saves to a new file.
    """
    try:
        # Load the CSV files
        df1 = pd.read_csv(source_file_1)
        df2 = pd.read_csv(source_file_2)

        # List of columns to transfer
        # Note: Ensure column names match exactly what is in the CSV header
        cols_to_transfer = ['J', 'J_ag', 'linesearch_step_size', 'MPC_t']

        # Check if columns exist in the second file
        missing_cols = [col for col in cols_to_transfer if col not in df2.columns]
        if missing_cols:
            print(f"Error: The following columns are missing in {source_file_2}: {missing_cols}")
            return

        # Update or add the columns in df1 with values from df2
        # We assume the rows align by index.
        for col in cols_to_transfer:
            df1[col] = df2[col]

        # Save to a new CSV file
        df1.to_csv(output_file, index=False)
        print(f"Successfully created {output_file}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Replace these filenames with your actual file paths
    name='CASE3.1'
    file_base = f'OriginalDATA\\{name}.csv'      # The base file
    file_source = f'RecreatedDATA\\{name}_cost.csv'    # The file containing the new column data
    file_output = f'{name}_final.csv'

    merge_specific_columns(file_base, file_source, file_output)