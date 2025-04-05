import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from modules.matching import InformationChecker
from modules.embedding import text_columns_to_vector
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from modules.model import HybridModel

# Load the data
df_account = pd.read_csv("account.csv")
df_profile = pd.read_csv("profile.csv")
df_description = pd.read_csv("description.csv")
df_passport = pd.read_csv("passport.csv")
df_labels = pd.read_csv("labels.csv")  # Make sure this has a 'label' column

categorical_cols = [
    "currency",
    "investment_experience",
    "type_of_mandate", "gender", "marital_status",
]

numeric_cols = [
   "inheritance_details", "real_estate_details"
]

# Creating the pipeline for initial checks
indices = np.random.choice(np.arange(0, 10000), size=2000, replace=False)

train_indx = indices[:1500]
test_indx = indices[1500:]

# checker = InformationChecker(df_passport, df_account, df_description, df_profile, df_labels)
# rejections = checker.create_mask()

# Combine by row index
df = pd.concat([df_account, df_profile, df_description, df_passport, df_labels], axis=1)
df = df.loc[:, ~df.columns.duplicated()]

df = pd.get_dummies(df, columns=categorical_cols)

df_output = pd.DataFrame()

# for rej_id in rejections:
#     if rej_id in test_indx:
#         df_output[f'Client_{rej_id}'] = 0

df_checked = df

print('-----------------', df_checked.columns)

cols = ['name', 'first_name', 'middle_name', 'last_name', 'passport_number',
       'address_city', 'address_street_name', 'address_street_number',
       'address_postal_code', 'country_of_domicile', 'phone_number',
       'email_address', 'address', 'birth_date', 'nationality',
       'passport_issue_date', 'passport_expiry_date', 'secondary_school',
       'higher_education', 'employment_history', 'aum', 'inheritance_details',
       'real_estate_details', 'investment_risk_profile', 'investment_horizon',
       'preferred_markets', 'secondary_school_name',
       'secondary_school_graduation_year', 'aum_savings', 'aum_inheritance',
       'aum_real_estate_value', 'inheritance_details_relationship',
       'inheritance_details_inheritance_year',
       'inheritance_details_profession',
       'inheritance_details_inheritance year', 'Summary Note',
       'Family Background', 'Education Background', 'Occupation History',
       'Wealth Summary', 'Client Summary', 'country', 'country_code',
       'passport_mrz', 'label']


# # Define features (X) and target (y)
train_df = df_checked[df_checked['client_id'].isin(train_indx)]
# print(f'*******', train_df.columns)
X = train_df.drop(columns=cols, inplace=False)  # Features (excluding 'target')
y = train_df['label']    # Target (the column to predict)

test_df = df_checked[df_checked['client_id'].isin(test_indx)]
X_test = test_df.drop(columns=cols, inplace=False)  # Features (excluding 'target')
y_test = test_df['label']    # Target (the column to predict)


# # ## Making prediction with the model
model  = HybridModel()
model.fit(X, y)
score = model.evaluate(X_test, y_test)
print(score)
print("hello")

# print(X.head())
# print(y.head())


# print(df_with_dummies.shape)
# print(df_with_dummies.columns)

# train_df = df_with_dummies.iloc[:5000]  
# print(train_df.columns)
# # # Use the next 500 rows as test set
# test_df = df_with_dummies.iloc[5000:6000]

# text_fields = ["Summary Note", "Family Background", "Education Background", "Occupation History", "Wealth Summary", "Client Summary"]

# # Creating the embedding for the descriptions
# #text_columns_to_vector(df_checked, text_fields)

# y = train_df['label']
# print(train_df.columns)
# x = train_df.drop(['label'])

# y_test = test_df['label']
# x_test = test_df.drop(['label'])

# print(df_with_dummies.shape)
# print(df_with_dummies.head())
# print(df_with_dummies.columns)