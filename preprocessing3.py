import pandas as pd
import re

# Load your merged DataFrame
df = pd.read_csv("updated_data.csv")

# Drop all columns that end with '_y'
df.drop(columns=[col for col in df.columns if col.endswith('_y')], inplace=True)

# Normalize remaining column names
def normalize_column_name(col):
    if col.endswith('_x'):
        col = col[:-2]  # Remove the '_x' suffix
    # Remove any numeric suffixes like .1, .2, etc., assuming they are followed by an underscore in some cases
    col = re.sub(r'\.\d+', '', col)
    return col

df.rename(columns=normalize_column_name, inplace=True)



# Save the cleaned DataFrame
df.to_csv("cleaned.csv", index=False)

print("All columns in the DataFrame:")
print(df.columns.tolist())
