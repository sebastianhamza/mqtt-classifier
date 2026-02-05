"""Semantic classification using spaCy word vectors."""

from typing import Dict, Tuple
from .constants import Category, CATEGORY_KEYWORDS


def _compute_vector_homogeneity(vectors) -> float:
    #Measure vector similarity
    if len(vectors) < 2:
        return 0.5
    
    import numpy as np
    
    #sample large vectors for performance
    if len(vectors) > 100:
        indices = np.linspace(0, len(vectors) - 1, 100, dtype=int)
        vectors = [vectors[i] for i in indices]
    
    similarities = []
    for i, v1 in enumerate(vectors):
        for v2 in vectors[i+1:]:
            sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
            similarities.append(sim)
    
    return sum(similarities) / len(similarities) if similarities else 0.5


def _compute_adaptive_temperature(text: str, doc) -> float:
    temp = 3.0
    
    if len(text) < 10:
        temp += 1.5
    elif len(text) > 500:
        temp -= 1.0
    
    vectors = [t.vector for t in doc if t.has_vector and not t.is_stop]
    if vectors:
        vector_similarity = _compute_vector_homogeneity(vectors)
        if vector_similarity > 0.7:
            temp += 0.5
        elif vector_similarity < 0.3:
            temp -= 0.5
    
    return max(1.0, min(8.0, temp))


def load_spacy_model(preferred: str = "en_core_web_md"):
    #Load spaCy model with fallback to 'en_core_web_sm'.
    try:
        import spacy
    except Exception:
        raise RuntimeError("spaCy not installed. Install with: pip install spacy")

    try:
        nlp = spacy.load(preferred)
        has_vectors = getattr(nlp, "vocab", None) is not None and nlp.vocab.vectors_length > 0
        return nlp, has_vectors
    except Exception:
        try:
            nlp = spacy.load("en_core_web_sm")
            return nlp, False
        except Exception:
            raise RuntimeError(
                "Failed to load spaCy model. Install a model like en_core_web_md:\n"
                "python -m spacy download en_core_web_md"
            )


def build_category_vectors(nlp) -> Dict[Category, object]:
    #Build mean vectors for each category from the seed terms.
    import numpy as _np
    cat_vecs = {}

    for category, terms in CATEGORY_KEYWORDS.items():
        vecs = []
        for t in terms:
            doc = nlp(t)
            if hasattr(doc, "vector") and len(doc.vector) > 0:
                vecs.append(doc.vector)

        if vecs:
            cat_vecs[category] = _np.mean(vecs, axis=0)

    return cat_vecs

def semantic_classify_multi(text: str, nlp, category_vectors, temperature: float = None, threshold: float = 0.20) -> Dict[Category, float]:
    import numpy as _np
    import math

    if not category_vectors:
        return {}

    # limit for semantic analysis
    MAX_SEMANTIC_LENGTH = 2000
    if len(text) > MAX_SEMANTIC_LENGTH:
        text = text[:MAX_SEMANTIC_LENGTH]

    doc = nlp(text)
    
    meaningful_vectors = [
        t.vector for t in doc 
        if not t.is_stop and not t.is_punct and t.has_vector
    ]

    if meaningful_vectors:
        vec = _np.mean(meaningful_vectors, axis=0)
    elif hasattr(doc, "vector") and len(doc.vector) > 0:
        vec = doc.vector
    else:
        return {}

    scores = {}
    for category, cvec in category_vectors.items():
        try:
            norm_v = _np.linalg.norm(vec)
            norm_c = _np.linalg.norm(cvec)
            if norm_v == 0 or norm_c == 0:
                score = 0.0
            else:
                score = float(_np.dot(vec, cvec) / (norm_v * norm_c))
        except Exception:
            score = 0.0
        scores[category] = score

    if not scores:
        return {}

    # bail out if everything looks the same (low confidence)
    max_sim = max(scores.values())
    min_sim = min(scores.values())
    sim_spread = max_sim - min_sim
    
    if max_sim < 0.15 or sim_spread < 0.04:
        return {}
    
    if len(text) < 8 and max_sim < 0.25:
        return {}

    if temperature is None:
        temperature = _compute_adaptive_temperature(text, doc)

    scaled_scores = {k: max(0, v) * temperature for k, v in scores.items()}
    
    max_raw = max(scaled_scores.values())
    exp_scores = {k: math.exp(v - max_raw) for k, v in scaled_scores.items()}
    total = sum(exp_scores.values())
    
    probs = {k: v / total for k, v in exp_scores.items()} if total > 0 else {k: 0.0 for k in scores}

    return {cat: conf for cat, conf in probs.items() if conf >= threshold}
