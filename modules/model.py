# model.py

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class HybridModel:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)