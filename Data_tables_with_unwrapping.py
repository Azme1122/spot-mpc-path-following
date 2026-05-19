import numpy as np
import pandas as pd
import os

name = 'CASE3.1_sim'
# Define the output directory
output_dir = os.path.join('Tables', name)
# output_dir = os.path.join('Comp', name)

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(f'{name}.csv')
# dir_input = 'OriginalDATA'
# df = pd.read_csv(os.path.join(dir_input, f'{name}.csv'))
df_new = df.columns.drop(['index','time_t'])

for col in df_new:
    output = df[['time_t',col]].copy()
    
    # Unwrap specific angle columns to avoid jumps in plots
    if col in ['theta_s', 'theta_s_m']:
        output[col] = np.unwrap(output[col])

    # output = df[[col]]
    output_filename = os.path.join(output_dir, f'{name}_{col}.tsv')
    output.to_csv(
        output_filename,
        sep='\t',
        header=False,
        index=False
    )