import numpy as np
import pandas as pd
import os

name = 'CASE1_unwrapped'
# Define the output directory
output_dir = os.path.join('Tables', name)
# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(f'CASE1.csv')
df_new = df.columns.drop(['index','time_t'])

for col in df_new:
    if col == 'theta_s':
        # unwrap theta_s
        theta_s = df['theta_s'].to_numpy()
        unwrapped_theta_s = np.unwrap(theta_s)
        output = pd.DataFrame({'time_t': df['time_t'], 'theta_s': unwrapped_theta_s})
        output_filename = os.path.join(output_dir, f'{name}_{col}.tsv')
        output.to_csv(
            output_filename,
            sep='\t',
            header=False,
            index=False
        )