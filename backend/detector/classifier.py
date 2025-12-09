import asyncio
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "clf.pkl")
DEBUG_DATA_PATH = os.path.join(MODEL_DIR, "debug_data.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Detector:
    def __init__(self, st_model: SentenceTransformer, clf, debug_data: Optional[Dict] = None):
        self.st_model = st_model
        self.clf = clf  # Can be any classifier with predict_proba
        self.debug_data = debug_data  # {'texts': [...], 'labels': [...], 'embeddings': [...]}

    async def embed(self, text: str) -> np.ndarray:
        # sentence-transformers is sync; run in thread to avoid blocking
        emb = await asyncio.to_thread(self.st_model.encode, [text], normalize_embeddings=True)
        return np.array(emb)[0]

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts at once for efficiency."""
        emb = await asyncio.to_thread(self.st_model.encode, texts, normalize_embeddings=True)
        return np.array(emb)

    async def predict(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"score": 0.0, "flag": False, "explanation": "Empty input"}
        emb = await self.embed(text)
        proba = await asyncio.to_thread(self.clf.predict_proba, emb.reshape(1, -1))
        ai_prob = float(proba[0][1])
        # Optional heuristics
        length = len(text.split())
        length_adj = 0.0
        if length > 80:
            length_adj = 0.03
        elif length < 5:
            length_adj = -0.03
        score = min(max(ai_prob + length_adj, 0.0), 1.0)
        
        # Determine model type for explanation
        clf_type = type(self.clf).__name__
        explanation = f"Ensemble score: {ai_prob:.2f}; model: {clf_type}; words: {length}"
        return {"score": score, "flag": score > 0.6, "explanation": explanation}

    async def predict_sentences(self, text: str) -> Dict[str, Any]:
        """
        Sentence-level detection for mixed human/AI content.
        Splits text into sentences and scores each one individually.
        """
        if not text or not text.strip():
            return {"overall_score": 0.0, "sentences": [], "summary": "Empty input"}
        
        # Split into sentences (handles ., !, ?, and line breaks)
        sentence_pattern = r'(?<=[.!?])\s+|\n+'
        raw_sentences = re.split(sentence_pattern, text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            # Fall back to treating entire text as one sentence
            sentences = [text.strip()]
        
        # Batch embed all sentences for efficiency
        embeddings = await self.embed_batch(sentences)
        
        # Get predictions for all sentences
        probas = await asyncio.to_thread(self.clf.predict_proba, embeddings)
        
        sentence_results = []
        ai_scores = []
        human_count = 0
        ai_count = 0
        
        for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
            ai_prob = float(probas[i][1])
            ai_scores.append(ai_prob)
            
            # Classify each sentence
            if ai_prob > 0.65:
                label = "AI"
                ai_count += 1
            elif ai_prob < 0.35:
                label = "Human"
                human_count += 1
            else:
                label = "Uncertain"
            
            sentence_results.append({
                "text": sent,
                "score": round(ai_prob, 3),
                "label": label
            })
        
        # Calculate overall metrics
        overall_score = float(np.mean(ai_scores))
        total = len(sentences)
        
        # Determine if mixed content
        if human_count > 0 and ai_count > 0:
            mix_status = f"MIXED: {human_count} human, {ai_count} AI, {total - human_count - ai_count} uncertain"
        elif ai_count == total:
            mix_status = "Fully AI-generated"
        elif human_count == total:
            mix_status = "Fully Human-written"
        else:
            mix_status = f"Mostly uncertain ({total - human_count - ai_count}/{total} sentences)"
        
        return {
            "overall_score": round(overall_score, 3),
            "flag": overall_score > 0.6,
            "total_sentences": total,
            "human_sentences": human_count,
            "ai_sentences": ai_count,
            "uncertain_sentences": total - human_count - ai_count,
            "summary": mix_status,
            "sentences": sentence_results
        }

    async def predict_debug(self, text: str) -> Dict[str, Any]:
        """Debug mode: returns prediction + most similar training samples."""
        if not text or not text.strip():
            return {"score": 0.0, "flag": False, "explanation": "Empty input", "similar_samples": []}
        
        emb = await self.embed(text)
        proba = await asyncio.to_thread(self.clf.predict_proba, emb.reshape(1, -1))
        ai_prob = float(proba[0][1])
        human_prob = float(proba[0][0])
        
        length = len(text.split())
        length_adj = 0.0
        if length > 80:
            length_adj = 0.03
        elif length < 5:
            length_adj = -0.03
        score = min(max(ai_prob + length_adj, 0.0), 1.0)
        
        # Find similar training samples
        similar_samples = []
        if self.debug_data is not None:
            train_embeddings = self.debug_data['embeddings']
            train_texts = self.debug_data['texts']
            train_labels = self.debug_data['labels']
            
            # Compute cosine similarities (embeddings are normalized)
            similarities = np.dot(train_embeddings, emb)
            top_indices = np.argsort(similarities)[-5:][::-1]  # Top 5 most similar
            
            for idx in top_indices:
                similar_samples.append({
                    "text": train_texts[idx][:200] + ("..." if len(train_texts[idx]) > 200 else ""),
                    "label": "Human" if train_labels[idx] == 0 else "AI",
                    "similarity": float(similarities[idx])
                })
        
        # Get individual model predictions if ensemble
        model_votes = {}
        if hasattr(self.clf, 'estimators_'):
            for name, estimator in zip(['LR', 'RF', 'SVM', 'GB', 'MLP'], self.clf.estimators_):
                est_proba = estimator.predict_proba(emb.reshape(1, -1))
                model_votes[name] = float(est_proba[0][1])
        
        return {
            "score": score,
            "flag": score > 0.6,
            "explanation": f"AI prob: {ai_prob:.3f}, Human prob: {human_prob:.3f}, words: {length}",
            "confidence": "high" if max(ai_prob, human_prob) > 0.8 else "medium" if max(ai_prob, human_prob) > 0.6 else "low",
            "model_votes": model_votes,
            "similar_samples": similar_samples
        }

async def get_detector() -> Detector:
    # Load embedding model
    st_model = await asyncio.to_thread(SentenceTransformer, EMBEDDING_MODEL_NAME)

    # Load debug data if available
    debug_data = None
    if os.path.exists(DEBUG_DATA_PATH):
        debug_data = await asyncio.to_thread(joblib.load, DEBUG_DATA_PATH)

    # Load or create classifier (supports ensemble or single model)
    if os.path.exists(MODEL_PATH):
        clf = await asyncio.to_thread(joblib.load, MODEL_PATH)
    else:
        # Minimal bootstrap training with tiny samples so POC runs out-of-the-box
        from sklearn.linear_model import LogisticRegression
        human_texts = [
            "I prepared by reviewing my past projects and reflecting on lessons learned.",
            "I prefer iterating quickly, gathering feedback, and improving incrementally.",
            "We faced a production outage and I coordinated with the team to resolve it.",
            "I don't know the answer yet, but I'd test and measure.",
        ]
        ai_texts = [
            "As an AI language model, I do not possess consciousness or intentions.",
            "The solution involves a holistic, scalable framework leveraging cutting-edge paradigms.",
            "Based on the provided context, here is a comprehensive step-by-step approach.",
            "Here is a detailed, structured response covering all facets of the problem.",
        ]
        X = await asyncio.to_thread(st_model.encode, human_texts + ai_texts, normalize_embeddings=True)
        y = np.array([0] * len(human_texts) + [1] * len(ai_texts))
        clf = LogisticRegression(max_iter=1000)
        await asyncio.to_thread(clf.fit, X, y)
        os.makedirs(MODEL_DIR, exist_ok=True)
        await asyncio.to_thread(joblib.dump, clf, MODEL_PATH)
    return Detector(st_model, clf, debug_data)
