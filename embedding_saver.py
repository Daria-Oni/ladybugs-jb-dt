import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from modules.matching import InformationChecker
from modules.embedding import text_columns_to_vector
from sklearn.compose import ColumnTransformer
from modules.model import HybridModel

df_description = pd.read_csv("description.csv")

df_description = df_description

text_fields = ["Summary Note", "Family Background", "Education Background", "Occupation History", "Wealth Summary", "Client Summary"]

# Creating the embedding for the descriptions
df = text_columns_to_vector(df_description, text_fields)

df.to_csv('description_embedded.csv')