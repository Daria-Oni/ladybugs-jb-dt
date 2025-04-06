import pandas as pd
import numpy as np
import xgboost as xgb
import ast
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from modules.matching import InformationChecker
from modules.embedding import text_columns_to_vector
from modules.parser import unzip_files, create_csvs
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from modules.model import HybridModel
from modules.preprocessing import classify_duration, calculate_education_level, calculate_year_diff
from sklearn.metrics import accuracy_score

# If not parsed parse the zip files
# unzip_files()
# create_csvs()

# # Creating the embedding for the descriptions
# text_fields = ["Summary Note", "Family Background", "Education Background", "Occupation History", "Wealth Summary", "Client Summary"]
# #text_columns_to_vector(df_checked, text_fields)

# Load the data
df_account = pd.read_csv("account.csv")
df_profile = pd.read_csv("profile.csv")
df_description = pd.read_csv("description.csv")
df_passport = pd.read_csv("passport.csv")
df_labels = pd.read_csv("labels.csv")  
df_embeddings = pd.read_csv("description_embedded.csv")


# Join the data from csv's
df = pd.concat([df_account, df_profile, df_description, df_passport, df_labels, df_embeddings], axis=1)
df = df.loc[:, ~df.columns.duplicated()]

# Fixing some features
df['investment_horizon'] = df['investment_horizon'].apply(classify_duration)
df['Total_Assets'] = df['aum_savings'] + df['aum_inheritance'] + df['aum_real_estate_value']
df['Age'] = pd.to_datetime('today').year - pd.to_datetime(df['birth_date']).dt.year

df['education_level'] = df['higher_education'].apply(lambda x: ast.literal_eval(x))
df['education_level'] = df['education_level'].apply(len)

df['employment_history'] = df['employment_history'].apply(lambda x: ast.literal_eval(x))
df['employment_duration'] = df['employment_history'].apply(calculate_year_diff)


# Feature Selection
print(df.columns)
categorical_col = ['gender', 'marital_status', 'investment_horizon', 'investment_experience', 'type_of_mandate']
feature_cols = ['Age', 'education_level',  'employment_duration', 'Total_Assets']
df = df[feature_cols+categorical_col+['client_id', 'label']]
df = pd.get_dummies(df, columns=categorical_col)

# print(df.columns)

# # embedded_columns = ['embedded_Summary Note',
# #        'embedded_Family Background', 'embedded_Education Background',
# #        'embedded_Occupation History', 'embedded_Wealth Summary',
# #        'embedded_Client Summary']

# # for col in embedded_columns:
# #     df[col] = df[col].apply(lambda x: x.strip())
#     # df[col] = df[col].apply(lambda x: x.replace(' ', ','))
    
#     # df[col] = df[col].apply(lambda x: np.array(ast.literal_eval(x)))

# # print(type(df.loc[0,'embedded_Client Summary']))
# # print(df.loc[0,'embedded_Client Summary'])

# # X = np.vstack(
# #     df.apply(lambda row: np.hstack([row[embedded_columns].values, row[feature_cols].values]), axis=1)
# # )

# # print(X.shape)  # Should be (num_rows, len(embedding) + len(other_cols))
# # print(X[0])

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
    if row['client_id'] in rejections:
        predictions.loc[index, 'label'] = 0
    else:
        predict = model.predict_proba(row[feature_cols].values.reshape(1, -1))[:, 1]
        # print(predict.shape)
        # print('****', predict)
        predictions.loc[index, 'label'] = (predict[0] > 0.5).astype(int)

score = accuracy_score(test_df['label'], predictions)
print(score)

# model  = HybridModel()
# model.fit(X_train.values, y_train.values)

# predictions = pd.DataFrame()

# for index, row in test_df.iterrows():
#     if row['client_id'] in rejections:
#         predictions.loc[index, 'label'] = 0
#     else:
#         predict = model.predict(row[feature_cols].values.reshape(1, -1))
#         predictions.loc[index, 'label'] = predict[0]

# print(predictions.shape)
