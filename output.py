import pandas as pd

df_account = pd.read_csv("account.csv")
df_profile = pd.read_csv("profile.csv")
df_description = pd.read_csv("description.csv")
df_passport = pd.read_csv("passport.csv")
df_labels = pd.read_csv("labels.csv")  



categorical_cols = [
    "currency",
    "investment_experience",
    "type_of_mandate", "gender", "marital_status",
]

# Join the data from csv's
df = pd.concat([df_account, df_profile, df_description, df_passport, df_labels], axis=1)
df.to_csv('data.csv', index=False)