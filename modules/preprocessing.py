import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from modules.matching import InformationChecker
from modules.embedding import text_columns_to_vector
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from modules.model import HybridModel
from sklearn.metrics import accuracy_score

df_account = pd.read_csv("account.csv")
df_profile = pd.read_csv("profile.csv")
df_description = pd.read_csv("description.csv")
df_passport = pd.read_csv("passport.csv")
df_labels = pd.read_csv("labels.csv")  
df_embeddings = pd.read_csv("description_embedded.csv")

categorical_cols = [
    "currency",
    "investment_experience",
    "type_of_mandate", "gender", "marital_status",
]

# Join the data from csv's
df = pd.concat([df_account, df_profile, df_description, df_passport, df_labels, df_embeddings], axis=1)
df = df.loc[:, ~df.columns.duplicated()]
df = pd.get_dummies(df, columns=categorical_cols)

print(df[df['aum_real_estate_value'].isna()])

# Function to classify durations
def classify_duration(duration):
    if "month" in duration:
        # Extract the number of months or range
        months = [int(s) for s in duration.split() if s.isdigit()]
        if len(months) == 1:
            month_count = months[0]
        elif len(months) == 2:
            month_count = (months[0] + months[1]) // 2  # Average for ranges
        else:
            month_count = 0  # Handle "Long-Term" etc.
        
        if month_count <= 1:
            return "Short"
        elif month_count <= 6:
            return "Medium"
        else:
            return "Long"
    
    elif "week" in duration:
        # Convert weeks to months (approx 4 weeks = 1 month)
        weeks = [int(s) for s in duration.split() if s.isdigit()]
        if len(weeks) == 1:
            month_count = weeks[0] / 4  # Convert weeks to months
        else:
            month_count = 0
        
        if month_count <= 1:
            return "Short"
        elif month_count <= 6:
            return "Medium"
        else:
            return "Long"
    else:
        return "Long"  # For cases like "Long-Term"