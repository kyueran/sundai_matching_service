# sundai/cosine_similarity.py
import numpy as np
from person_vectorizer import Person, PersonVectorizer

def cosine_similarity_q1_q2_flipped(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Compute cosine similarity between two people’s vectors,
    but for vector2, swap the first and second fields before comparison.

    vector1: np.ndarray of shape (num_fields, dim)
    vector2: np.ndarray of shape (num_fields, dim)

    Returns:
        cosine similarity score (float)
    """
    # defensive copy to avoid modifying input
    v2 = vector2.copy()

    # swap field 0 and 1 in vector2
    if v2.shape[0] >= 2:
        v2[[0, 1]] = v2[[1, 0]]

    # flatten both for similarity (combine fields into one vector)
    v1_flat = v1 = vector1.flatten()
    v2_flat = v2.flatten()

    # normalize
    norm1 = np.linalg.norm(v1_flat)
    norm2 = np.linalg.norm(v2_flat)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1_flat, v2_flat) / (norm1 * norm2))
