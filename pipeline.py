import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from modules.matching import InformationChecker
from modules.embedding import text_columns_to_vector
from sklearn.compose import ColumnTransformer
from modules.model import HybridModel

# Load the data
df_account = pd.read_csv("account.csv")
df_profile = pd.read_csv("profile.csv")
df_description = pd.read_csv("description.csv")
df_passport = pd.read_csv("passport.csv")
df_labels = pd.read_csv("labels.csv")  # Make sure this has a 'label' column

# Creating the pipeline for initial checks
checker = InformationChecker(df_passport, df_account, df_description, df_profile, df_labels)
rejections = checker.create_mask()

# Combine by row index
df = pd.concat([df_account, df_profile, df_description, df_passport, df_labels], axis=1)
df = df.loc[:, ~df.columns.duplicated()]

print(df.shape)
df_checked = df[~df['client_id'].isin(rejections)]
print(df_checked.shape)
# TODO: Actually remove the wrong ones

# Column name for labels (adjust if needed)
label_column = "label"

# Sample 500 rows with label 0 and 500 with label 1
#train_label_0 = df[df[label_column] == 0].sample(n=500, random_state=42)
#train_label_1 = df[df[label_column] == 1].sample(n=500, random_state=42)

# Combine and shuffle the training set
#train_df = pd.concat([train_label_0, train_label_1]).sample(frac=1, random_state=42)
train_df = df_checked.iloc[:1000]  

# # Use the next 500 rows as test set
test_df = df_checked.iloc[1000:1500]

text_fields = ["Summary Note", "Family Background", "Education Background", "Occupation History", "Wealth Summary", "Client Summary"]

# Creating the embedding for the descriptions
# text_columns_to_vector(df_checked, text_fields)

categorical_cols = [
    "currency",
    "investment_experience",
    "type_of_mandate", "gender", "marital_status",
]

numeric_cols = [
   "inheritance_details", "real_estate_details"
]

y = df_checked['label']
df_checked  = df_checked[categorical_cols]
df_with_dummies = pd.get_dummies(df_checked, columns=categorical_cols)


y_test = test_df['label']
test_df = test_df[categorical_cols]
test_df_dummy = pd.get_dummies(test_df, columns=categorical_cols)

print(df_with_dummies.shape)
print(df_with_dummies.head())
print(df_with_dummies.columns)




# ## Making prediction with the model
model  = HybridModel()
model.fit(df_with_dummies, y)
score = model.evaluate(test_df_dummy, y_test)
print(score)