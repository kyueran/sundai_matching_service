from dataclasses import dataclass
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class Person:
    username: str
    name: str
    linkedin: str
    role: str
    q1: str
    q2: str
    q3: str
    q4: str

# person_vectorizer.py

class PersonVectorizer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_person_fields(self, person: Person) -> np.ndarray:
        texts = [person.q1, person.q2, person.q3, person.q4]
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)

    def encode_person_weighted(self, person: Person, weights=None) -> np.ndarray:
        vecs = self.encode_person_fields(person)
        if weights is None:
            weights = np.array([0.4, 0.4, 0.1, 0.1], dtype=np.float32)
        weights = weights / np.sum(weights)
        weighted_vec = np.average(vecs, axis=0, weights=weights)
        return weighted_vec
