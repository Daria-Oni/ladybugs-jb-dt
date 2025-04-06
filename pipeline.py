import pandas as pd
import numpy as np
import xgboost as xgb
import ast
import json
import re
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

# If not parsed parse the zip files
# unzip_files()
# create_csvs()

# Load the data
df_account = pd.read_csv("account.csv")
df_profile = pd.read_csv("profile.csv")
df_description = pd.read_csv("description.csv")
df_passport = pd.read_csv("passport.csv")
df_labels = pd.read_csv("labels.csv")  

# # Join the data from csv's
df = pd.concat([df_account, df_profile, df_description, df_passport, df_labels], axis=1)
df = df.loc[:, ~df.columns.duplicated()]

categorical_col = ['gender', 'marital_status', 'investment_horizon', 'investment_experience', 'type_of_mandate']
feature_cols = ['Age', 'education_level',  'employment_duration', 'Total_Assets_scaled', 'has_switzerland']


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



df = df[feature_cols+categorical_col+['client_id', 'label']]
df = pd.get_dummies(df, columns=categorical_col)

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Check the matching information
checker = InformationChecker(df_passport, df_account, df_description, df_profile, df_labels)
rejections = checker.create_mask()

train_df_checked = train_df[~train_df['client_id'].isin(rejections)] # This is the dataset we feed to the model

label_col = 'label'

X_train = train_df_checked[feature_cols]
y_train = train_df_checked[label_col]

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Step 4: Train the model
model.fit(X_train.values, y_train.values)
predictions = pd.DataFrame()

for index, row in test_df.iterrows():
    predictions.loc[index, 'client_id'] = f'client_{row['client_id']}'
    if row['client_id'] in rejections:
        predictions.loc[index, 'label'] = 0
    else:
        predict = model.predict_proba(row[feature_cols].values.reshape(1, -1))[:, 1]
        predictions.loc[index, 'label'] = (predict[0] > 0.5).astype(int)

score = accuracy_score(test_df['label'], predictions['label'])
print(score)

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