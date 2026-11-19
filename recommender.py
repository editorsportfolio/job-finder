"""Retrieval (embeddings) -> ranking (LightGBM). Plus a popularity baseline."""
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from data import ALL_SKILLS

FEATURES = ["sim", "overlap", "level_match", "city_match", "days_old", "popularity"]


class Embedder:
    """Uses sentence-transformers if it's installed, else TF-IDF + SVD (works offline)."""

    def __init__(self, texts):
        self.model = None
        if os.getenv("EMBEDDER") != "tfidf":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                pass
        if self.model is None:
            self.tfidf = TfidfVectorizer().fit(texts)
            dims = min(48, len(self.tfidf.vocabulary_) - 1)
            self.svd = TruncatedSVD(dims, random_state=0).fit(self.tfidf.transform(texts))
        self.name = "MiniLM" if self.model else "tfidf+svd"

    def encode(self, texts):
        if self.model is not None:
            return self.model.encode(list(texts), normalize_embeddings=True)
        return normalize(self.svd.transform(self.tfidf.transform(texts)))


def find_skills(text):
    """Dumb-but-effective skill extraction: keyword match."""
    text = text.lower()
    return {s for s in ALL_SKILLS if s in text}


class Recommender:
    def __init__(self, jobs, users, clicks, rank_from=30, rank_to=45):
        self.jobs = jobs.reset_index(drop=True)
        self.emb = Embedder(self.jobs.title + " " + self.jobs.description)
        self.job_vecs = self.emb.encode(self.jobs.title + " " + self.jobs.description)
        self.job_skills = [set(s.split()) for s in self.jobs.skills]

        # popularity from the early period only, so it can't leak the labels we train on
        early = clicks[clicks.day < rank_from]
        counts = early.groupby("job_id").size().reindex(self.jobs.job_id, fill_value=0)
        self.pop = np.log1p(counts.values)

        self._train(users, clicks[(clicks.day >= rank_from) & (clicks.day < rank_to)])

    # ---- retrieval ----
    def candidates(self, resume, k=100):
        user_vec = self.emb.encode([resume])[0]
        sims = self.job_vecs @ user_vec
        top = np.argsort(-sims)[:k]
        return top, sims[top]

    # ---- features ----
    def features(self, idx, sims, skills, level, city):
        j = self.jobs.iloc[idx]
        overlap = [len(skills & self.job_skills[i]) / len(self.job_skills[i]) for i in idx]
        return pd.DataFrame({
            "sim": sims,
            "overlap": overlap,
            "level_match": (j.level.values == level).astype(int),
            "city_match": (j.city.values == city).astype(int),
            "days_old": j.days_old.values,
            "popularity": self.pop[idx],
        })

    # ---- training ----
    def _train(self, users, rank_clicks):
        X, y, groups = [], [], []
        for _, u in users.iterrows():
            idx, sims = self.candidates(u.resume)
            seen = rank_clicks[rank_clicks.user_id == u.user_id].set_index("job_id").label
            labels = [int(seen.get(self.jobs.job_id[i], 0)) for i in idx]
            X.append(self.features(idx, sims, find_skills(u.resume), u.level, u.city))
            y += labels
            groups.append(len(idx))
        self.ranker = LGBMRanker(n_estimators=150, learning_rate=0.05, num_leaves=15, verbose=-1)
        self.ranker.fit(pd.concat(X), y, group=groups)

    # ---- recommending ----
    def recommend(self, resume, level, city, k=10, exclude=(), method="ranked"):
        idx, sims = self.candidates(resume)
        if method == "popularity":
            order = np.argsort(-self.pop)
            idx = np.array([i for i in order if self.jobs.job_id[i] not in exclude])[:k]
            return idx, self.pop[idx]
        keep = np.array([self.jobs.job_id[i] not in exclude for i in idx])
        idx, sims = idx[keep], sims[keep]
        if method == "embeddings":
            return idx[:k], sims[:k]
        f = self.features(idx, sims, find_skills(resume), level, city)
        scores = self.ranker.predict(f)
        best = np.argsort(-scores)[:k]
        return idx[best], scores[best]
