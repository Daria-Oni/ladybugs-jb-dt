import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

def get_bert_embedding(text, tokenizer, model):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embedding of [CLS] token (sentence-level representation)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [1, 768]
    
    return cls_embedding.squeeze().numpy()  # Convert to 1D numpy array

def text_columns_to_vector(df, column_list):
    # Load pre-trained BERT base model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # Set to evaluation mode

    df = df.copy()
    for col in column_list:
        df.loc[:, f'embedded_{col}'] = df[col].apply(lambda x: get_bert_embedding(str(x), tokenizer, model))
    return df
        
# # Load your data
# df1 = pd.read_csv("account.csv")
# df2 = pd.read_csv("profile.csv")
# df4 = pd.read_csv("passport.csv")

# # Combine by row index
# df = pd.concat([df1, df2, df3, df4], axis=1)
# df = df.loc[:, ~df.columns.duplicated()]

# # ---- STEP 1: Drop unnecessary fields ----
# drop_cols = [
#     "name", "first_name", "middle_name", "last_name",
#     "passport_number", "passport_mrz", "address_street_name"
# ]
# df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# categorical_cols = [
#     "currency", "country", "country_code",
#     "investment_risk_profile", "investment_horizon", "investment_experience",
#     "type_of_mandate", "gender", "marital_status", "nationality", "country_of_domicile"
# ]

# numeric_cols = [
# #    "inheritance_details", "real_estate_details"
# ]

# # ---- STEP 3: Prepare text input for embeddings ----
# df[text_fields] = df[text_fields].fillna("")
# text_inputs = df[text_fields].apply(lambda row: "\n".join(row.values.astype(str)), axis=1).tolist()

# # ---- STEP 4: Prepare tabular features ----
# tabular_data = df[numeric_cols + categorical_cols].copy()

# # Preprocessing pipeline
# preprocessor = ColumnTransformer(transformers=[
#     ("num", StandardScaler(), numeric_cols),
#     ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
# ])
# X_tabular = preprocessor.fit_transform(tabular_data)  # shape: (N, D1)

# # ---- STEP 5: Embed text with NV-Embed ----
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_id = "nvidia/NV-Embed-v2"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModel.from_pretrained(model_id).to(device).eval()

# def embed_texts(texts, batch_size=16):
#     all_embeddings = []
#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i+batch_size]
#         inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             embeddings = outputs.last_hidden_state[:, 0]  # CLS token
#             embeddings = normalize(embeddings, p=2, dim=1)
#         all_embeddings.append(embeddings.cpu().numpy())
#     return np.vstack(all_embeddings)

# X_text = embed_texts(text_inputs)  # shape: (N, 1024)
# print(X_text)
# print("hello")

# # ---- STEP 6: Combine into hybrid vector ----
# X_hybrid = np.hstack([X_tabular, X_text])  # shape: (N, D1 + 1024)
