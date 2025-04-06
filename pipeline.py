import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from modules.matching import InformationChecker
from modules.embedding import text_columns_to_vector
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from modules.model import HybridModel
from modules.preprocessing import classify_duration
from sklearn.metrics import accuracy_score

# If not parsed parse the zip files

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


categorical_cols = [
    "currency",
    "investment_experience",
    "investment_risk_profile",
    "investment_horizon",
    "type_of_mandate", "gender", "marital_status",
]

# Join the data from csv's
df = pd.concat([df_account, df_profile, df_description, df_passport, df_labels, df_embeddings], axis=1)


# Fixing some features
df['investment_horizon'] = df['investment_horizon'].apply(classify_duration)

df = df.loc[:, ~df.columns.duplicated()]
df = pd.get_dummies(df, columns=categorical_cols)



# embedded_columns = ['embedded_Summary Note',
#        'embedded_Family Background', 'embedded_Education Background',
#        'embedded_Occupation History', 'embedded_Wealth Summary',
#        'embedded_Client Summary']

feature_cols = ['aum_savings', 'aum_inheritance',
       'aum_real_estate_value', 'currency_CHF', 'currency_DKK',
       'investment_horizon_Short', 'investment_horizon_Medium', 'investment_horizon_Long',
       'currency_EUR', 'investment_experience_Experienced',
       'investment_experience_Expert', 'investment_experience_Inexperienced',
       'type_of_mandate_Advisory', 'type_of_mandate_Discretionary',
       'type_of_mandate_Execution-Only', 'type_of_mandate_Hybrid', 'gender_F',
       'gender_M', 'marital_status_divorced', 'marital_status_married',
       'marital_status_single', 'marital_status_widowed']

# for col in embedded_columns:
#     df[col] = df[col].apply(lambda x: x.strip())
    # df[col] = df[col].apply(lambda x: x.replace(' ', ','))
    
    # df[col] = df[col].apply(lambda x: np.array(ast.literal_eval(x)))

# print(type(df.loc[0,'embedded_Client Summary']))
# print(df.loc[0,'embedded_Client Summary'])

# X = np.vstack(
#     df.apply(lambda row: np.hstack([row[embedded_columns].values, row[feature_cols].values]), axis=1)
# )

# print(X.shape)  # Should be (num_rows, len(embedding) + len(other_cols))
# print(X[0])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Check the matching information
checker = InformationChecker(df_passport, df_account, df_description, df_profile, df_labels)
rejections = checker.create_mask()

train_df_checked = train_df[~train_df['client_id'].isin(rejections)] # This is the dataset we feed to the model

# print(train_df_checked.columns)

# ['name', 'first_name', 'middle_name', 'last_name', 'passport_number',
#        'address_city', 'address_street_name', 'address_street_number',
#        'address_postal_code', 'country_of_domicile', 'phone_number',
#        'email_address', 'address', 'client_id', 'birth_date', 'nationality',
#        'passport_issue_date', 'passport_expiry_date', 'secondary_school',
#        'higher_education', 'employment_history', 'aum', 'inheritance_details',
#        'real_estate_details', 'investment_risk_profile', 'investment_horizon',
#        'preferred_markets', 'secondary_school_name',
#        'secondary_school_graduation_year', 'aum_savings', 'aum_inheritance',
#        'aum_real_estate_value', 'inheritance_details_relationship',
#        'inheritance_details_inheritance_year',
#        'inheritance_details_profession',
#        'inheritance_details_inheritance year', 'Summary Note',
#        'Family Background', 'Education Background', 'Occupation History',
#        'Wealth Summary', 'Client Summary', 'country', 'country_code',
#        'passport_mrz', 'label', 'Unnamed: 0', 'embedded_Summary Note',
#        'embedded_Family Background', 'embedded_Education Background',
#        'embedded_Occupation History', 'embedded_Wealth Summary',
#        'embedded_Client Summary', 'currency_CHF', 'currency_DKK',
#        'currency_EUR', 'investment_experience_Experienced',
#        'investment_experience_Expert', 'investment_experience_Inexperienced',
#        'type_of_mandate_Advisory', 'type_of_mandate_Discretionary',
#        'type_of_mandate_Execution-Only', 'type_of_mandate_Hybrid', 'gender_F',
#        'gender_M', 'marital_status_divorced', 'marital_status_married',
#        'marital_status_single', 'marital_status_widowed']


label_col = 'label'

X_train = train_df_checked[feature_cols]
y_train = train_df_checked[label_col]

model  = HybridModel()
model.fit(X_train.values, y_train.values)

predictions = pd.DataFrame()

for index, row in test_df.iterrows():
    if row['client_id'] in rejections:
        predictions.loc[index, 'label'] = 0
    else:
        predict = model.predict(row[feature_cols].values.reshape(1, -1))
        predictions.loc[index, 'label'] = predict[0]

print(predictions.shape)
score = accuracy_score(test_df['label'], predictions)
print(score)