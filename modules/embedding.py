import os
import zipfile
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
import torch

# import nltk
# nltk.download('punkt_tab')

import re

def tokenize(text):
    # Lowercase and extract words/numbers
    return re.findall(r'\b\w+\b', text.lower())

def get_bert_embedding(text, tokenizer, model):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embedding of [CLS] token (sentence-level representation)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [1, 768]
    
    return cls_embedding.squeeze().numpy()  # Convert to 1D numpy array

def get_average_embedding(text, tokenizer, model):
    sentences = sent_tokenize(text)  # Split text into sentences
    embeddings = []
    
    for sentence in sentences:
        embedding = get_bert_embedding(sentence, tokenizer, model)
        embeddings.append(embedding)
    
    # Average the sentence embeddings to get one final embedding for the text
    return np.mean(embeddings, axis=0)

def text_columns_to_vector(df, column_list):
    # Load pre-trained BERT base model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # Set to evaluation mode

    df = df.copy()
    for col in column_list:
        df.loc[:, f'embedded_{col}'] = df[col].apply(lambda x: get_average_embedding(str(x), tokenizer, model))
    return df

# Function to get the average Word2Vec vector for a sentence
def get_average_word2vec(description, model):
    words = description.split()
    vectors = []
    for word in words:
        if word in model.key_to_index:  # Ensure the word exists in the model's vocabulary
            vectors.append(model[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)  # Return a zero vector if no words are in the model

def load_glove_vectors(zip_file):
    zip_file_path = zip_file # Replace with the path to your zip file
    extract_dir = 'glove_files/'  # Replace with the directory where you want to extract files

    # Create the extraction directory if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)

        # Unzip the file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        print(f'Files extracted to {extract_dir}')

    glove_vectors = {}
    with open(os.path.join(extract_dir, 'glove.42B.300d.txt'), 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_vectors[word] = vector
    return glove_vectors

# Function to get the average GloVe vector for a sentence
def get_glove_embedding(description, glove_vectors):
    tokens = tokenize(description)
    vectors = [glove_vectors[word] for word in tokens if word in glove_vectors]
    
    for word in vectors:
        word = word.lower()  # Normalize to lowercase
        if word in glove_vectors:  # Check if word is in the GloVe dictionary
            vectors.append(glove_vectors[word])
    
    if vectors:
        # Return the mean of the vectors, or a zero vector if no word is in GloVe
        return np.mean(vectors, axis=0)
    else:
        # Return a zero vector if no valid words
        return np.zeros(50)  # The dimension should match the GloVe vector size (50 in this case)
