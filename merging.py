import pandas as pd

# Load the CSV files
df1 = pd.read_csv('rejections.csv', header=None)   # Update path
df2 = pd.read_csv('rejected_clients.csv', header=None)  # Update path
df3 = pd.read_csv('rejected_clients_acc_prof.csv', header=None)   # Update path

# Assign column names (assuming 'client_id' is the first column and 'prediction' is the second)
column_names = ['client_id', 'prediction']
df1.columns = df2.columns = df3.columns = column_names

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Remove duplicates based on 'client_id' column
combined_df = combined_df.drop_duplicates(subset=['client_id'])


combined_df.to_csv('merging.csv', index=False)