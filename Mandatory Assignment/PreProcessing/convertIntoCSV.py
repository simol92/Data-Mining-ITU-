import pandas as pd

# Read the TSV file
data = pd.read_csv('original.tsv', sep='\t')

# Write the data to a CSV file
data.columns = data.columns.str.strip()

data.to_csv('data.csv', index=False)
