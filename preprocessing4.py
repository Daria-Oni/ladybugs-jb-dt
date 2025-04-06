import pandas as pd

# Load your DataFrame
df = pd.read_csv("cleaned.csv")


# Function to normalize column names and remove duplicates
def remove_duplicate_columns(df):
    # Create a dictionary to keep track of the first occurrence of each normalized column name
    columns_seen = {}
    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Normalize the column name (remove numerical suffixes like .1, .2, ...)
        normalized_column = column.split('.')[0]
        # If the normalized name has been encountered before, mark the column for deletion
        if normalized_column in columns_seen:
            df.drop(column, axis=1, inplace=True)
        else:
            # If it hasn't been encountered, add it to the dictionary
            columns_seen[normalized_column] = True

    return df

# Apply the function
df = remove_duplicate_columns(df)

# Save the cleaned DataFrame
df.to_csv("noduplicates.csv", index=False)

# Print the columns after removing duplicates to verify
print("Columns after removing duplicates based on normalized names:")
print(df.columns.tolist())

