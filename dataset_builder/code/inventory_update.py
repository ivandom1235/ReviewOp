from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from bertopic import BERTopic
except ImportError:
    BERTopic = None

def run_bertopic_update(texts: List[str], existing_aspects: List[str], threshold: float = 0.6) -> List[Dict[str, Any]]:
    """
    1. Run BERTopic over new unlabeled reviews
    2. Compare new topics against existing inventory using cosine similarity
    3. Topics with max similarity < threshold are flagged as candidate_new_aspect
    """
    if BERTopic is None:
        raise ImportError("BERTopic is not installed. Please install it using `pip install bertopic`")
        
    if len(texts) < 50:
        return [] # Too few texts to form meaningful topics
        
    topic_model = BERTopic(min_topic_size=max(10, len(texts) // 100))
    topics, _ = topic_model.fit_transform(texts)
    
    topic_info = topic_model.get_topic_info()
    new_candidate_aspects = []
    
    encoder = SentenceTransformer("all-mpnet-base-v2")
    existing_embeddings = encoder.encode(existing_aspects)
    
    for i, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1: 
            # -1 is for outliers in BERTopic
            continue 
            
        topic_words = topic_model.get_topic(topic_id)
        # Take the top 3 words as candidate representation
        top_words = " ".join([word for word, _ in topic_words[:3]])
        
        candidate_embed = encoder.encode([top_words])
        
        # Calculate cosine similarity manually using numpy dot product
        sims = np.dot(candidate_embed, existing_embeddings.T) / (
            np.linalg.norm(candidate_embed) * np.linalg.norm(existing_embeddings, axis=1)
        )
        max_sim = float(np.max(sims))
        
        if max_sim < threshold:
            example_idx = [idx for idx, t_id in enumerate(topics) if t_id == topic_id]
            example_texts = [texts[idx] for idx in example_idx[:5]]
            
            new_candidate_aspects.append({
                "proposed_name": top_words,
                "similarity_to_closest_existing": max_sim,
                "example_sentences": example_texts,
                "document_count": int(row['Count'])
            })
            
    return new_candidate_aspects
