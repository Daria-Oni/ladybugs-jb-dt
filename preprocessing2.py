import pandas as pd

def load_data():
    """Load data with proper error handling."""
    try:
        df_data = pd.read_csv('data.csv')
        df_merging = pd.read_csv('merging.csv', header=None, names=['client_id', 'prediction'])
        return df_data, df_merging
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None

def remove_duplicates(df_data, df_merging):
    """Remove duplicates based on 'client_id'."""
    common = pd.merge(df_data, df_merging, on=['client_id'], how='inner')
    if not common.empty:
        print("Rows that will be removed:")
        print(common)

    df_data_cleaned = df_data.merge(common, on=['client_id'], how='left', indicator=True)
    df_data_cleaned = df_data_cleaned[df_data_cleaned['_merge'] == 'left_only']
    df_data_cleaned.drop(columns=['_merge'], inplace=True)

    return df_data_cleaned


def main():
    df_data, df_merging = load_data()
    if df_data is not None and df_merging is not None:
        df_data_cleaned = remove_duplicates(df_data, df_merging)
        print("Updated column names:", df_data_cleaned.columns)
        df_data_cleaned.to_csv('updated_data.csv', index=False)
        print("Data has been cleaned and saved.")

if __name__ == "__main__":
    main()

