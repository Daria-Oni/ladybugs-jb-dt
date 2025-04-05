import pandas as pd

# Load the data
df1 = pd.read_csv("account.csv")
df2 = pd.read_csv("profile.csv")
df3 = pd.read_csv("description.csv")
df4 = pd.read_csv("passport.csv")
df5 = pd.read_csv("labels.csv")  # Make sure this has a 'label' column

# Combine by row index
df = pd.concat([df1, df2, df3, df4, df5], axis=1)
df = df.loc[:, ~df.columns.duplicated()]

# Column name for labels (adjust if needed)
label_column = "label"

# Sample 500 rows with label 0 and 500 with label 1
#train_label_0 = df[df[label_column] == 0].sample(n=500, random_state=42)
#train_label_1 = df[df[label_column] == 1].sample(n=500, random_state=42)

# Combine and shuffle the training set
#train_df = pd.concat([train_label_0, train_label_1]).sample(frac=1, random_state=42)
train_df = df.iloc[:1000]  
# Use the next 500 rows as test set
test_df = df.iloc[1000:1500]

