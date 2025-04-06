import pandas as pd
import numpy as np
import xgboost as xgb
import ast
import json
import re
import os
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from modules.matching import InformationChecker
from modules.embedding import text_columns_to_vector, load_glove_vectors, get_glove_embedding
from modules.parser import unzip_files, create_csvs
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from modules.model import HybridModel
from modules.preprocessing import classify_duration, calculate_education_level, calculate_year_diff
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer

current_directory = os.getcwd()
print(current_directory)
# Unzipping the client files and creating the csv
extract_to = f'{current_directory}/train'
os.makedirs(extract_to, exist_ok=True)
# If not parsed parse the zip files
train_path = os.path.join(current_directory, 'train')
unzip_files(extract_to)
create_csvs(train_path)


extract_to = f'{current_directory}/test'
os.makedirs(extract_to, exist_ok=True)
# If not parsed parse the zip files
test_path = os.path.join(current_directory, 'test')
unzip_files(extract_to)
create_csvs(test_path)

# Load the Training Data
df_account = pd.read_csv(f"{train_path}/account.csv")
df_profile = pd.read_csv(f"{train_path}/profile.csv")
df_description = pd.read_csv(f"{train_path}/description.csv")
df_passport = pd.read_csv(f"{train_path}/passport.csv")
df_labels = pd.read_csv(f"{train_path}/labels.csv")  

# # Join the data from csv's
df = pd.concat([df_account, df_profile, df_description, df_passport, df_labels], axis=1)
df = df.loc[:, ~df.columns.duplicated()]

categorical_col = ['gender', 'marital_status', 'investment_horizon', 'investment_experience', 'type_of_mandate']
feature_cols = ['Age', 'education_level',  'employment_duration', 'Total_Assets_scaled', 'has_switzerland']


def apply_processing(data_frame, test_mode=False):
    df = data_frame.copy()
    df['investment_horizon'] = df['investment_horizon'].apply(classify_duration)
    df['Total_Assets'] = df['aum_savings'] + df['aum_inheritance'] + df['aum_real_estate_value']
    df['Total_Assets'] = df.apply(
        lambda row: row['Total_Assets'] * 1.06 if row['currency'] == 'CHF' 
        # else row['Total_Assets'] * 0.13 if row['currency'] == 'DKK' 
        else row['Total_Assets'],
        axis=1
    )
    scaler = StandardScaler()
    # Fit and transform the numeric column (feature1) and add it to the dataframe
    df['Total_Assets_scaled'] = scaler.fit_transform(df[['Total_Assets']])

    def parse_country_list(country_list_str):
        return ast.literal_eval(country_list_str)

    # Apply the function to parse the 'preferred_markets' column
    df['preferred_markets'] = df['preferred_markets'].apply(parse_country_list)
    df['Age'] = pd.to_datetime('today').year - pd.to_datetime(df['birth_date']).dt.year

    df['education_level'] = df['higher_education'].apply(lambda x: ast.literal_eval(x))
    df['education_level'] = df['education_level'].apply(len)

    df['employment_history'] = df['employment_history'].apply(lambda x: ast.literal_eval(x))
    df['employment_duration'] = df['employment_history'].apply(calculate_year_diff)
    df['has_switzerland'] = df['preferred_markets'].apply(lambda x: 1 if 'Switzerland' in x else 0)

    if not test_mode:
        df = df[feature_cols+categorical_col+['client_id', 'label']]
    else:
        df = df[feature_cols+categorical_col+['client_id']]
    df = pd.get_dummies(df, columns=categorical_col)
    return df

df = apply_processing(df)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Check the matching information
checker = InformationChecker(df_passport, df_account, df_description, df_profile, df_labels)
rejections = checker.create_mask()

# train_df_checked = train_df[~train_df['client_id'].isin(rejections)] # This is the dataset we feed to the model
train_df_checked = df[~df['client_id'].isin(rejections)] # This is the dataset we feed to the model

label_col = 'label'

X_train = train_df_checked[feature_cols]
y_train = train_df_checked[label_col]

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Step 4: Train the model
model.fit(X_train.values, y_train.values)

# Load the test data
# Load the Training Data
df_account_t = pd.read_csv(f"{test_path}/account.csv")
df_profile_t = pd.read_csv(f"{test_path}/profile.csv")
df_description_t = pd.read_csv(f"{test_path}/description.csv")
df_passport_t = pd.read_csv(f"{test_path}/passport.csv")

# # Join the data from csv's
df_test = pd.concat([df_account_t, df_profile_t, df_description_t, df_passport_t], axis=1)
df_test = df_test.loc[:, ~df_test.columns.duplicated()]

df_test = apply_processing(df_test, test_mode=True)
# Check the matching information
checker = InformationChecker(df_passport_t, df_account_t, df_description_t, df_profile_t, df_labels, True)
rejections = checker.create_mask()

predictions = pd.DataFrame()

for index, row in df_test.iterrows():
    predictions.loc[index, 'client_id'] = f'client_{row['client_id']}'
    if row['client_id'] in rejections:
        predictions.loc[index, 'label'] = 0
    else:
        predict = model.predict_proba(row[feature_cols].values.reshape(1, -1))[:, 1]
        predictions.loc[index, 'label'] = (predict[0] > 0.5).astype(int)

# score = accuracy_score(test_df['label'], predictions['label'])
# print(score)

predictions['label'] = predictions['label'].map({0: 'Reject', 1: 'Accept'})
predictions.to_csv('ladybug.csv', header=False, index=False, sep=';')


# model  = HybridModel()
# model.fit(X_train.values, y_train.values)

# predictions = pd.DataFrame()

# for index, row in test_df.iterrows():
#     if row['client_id'] in rejections:
#         predictions.loc[index, 'label'] = 0
#     else:
#         predict = model.predict_proba(row[feature_cols].values.reshape(1, -1))[:, 1]
#         predictions.loc[index, 'label'] = (predict[0] > 0.5).astype(int)

# print(predictions.shape)
# score = accuracy_score(test_df['label'], predictions)
# print(score)