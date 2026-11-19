"""Compares popularity vs embeddings vs embeddings+LightGBM ranker on held-out clicks."""
import numpy as np

from data import make_jobs, make_users, make_clicks
from recommender import Recommender

K = 10


def precision_at_k(recs, relevant, k=K):
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs, relevant, k=K):
    return len(set(recs[:k]) & relevant) / len(relevant)


def ndcg_at_k(recs, relevant, k=K):
    gains = [1 / np.log2(i + 2) for i, r in enumerate(recs[:k]) if r in relevant]
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return sum(gains) / ideal


def main():
    jobs, users = make_jobs(), make_users()
    clicks = make_clicks(users, jobs)
    rec = Recommender(jobs, users, clicks)
    print("embedder:", rec.emb.name)

    # train period = day < 45, test period = day >= 45 (only clicks/saves count as relevant)
    past = clicks[clicks.day < 45]
    test = clicks[(clicks.day >= 45) & (clicks.label >= 2)]

    results = {m: {"P@10": [], "R@10": [], "NDCG@10": []} for m in ["popularity", "embeddings", "ranked"]}
    for _, u in users.iterrows():
        relevant = set(test[test.user_id == u.user_id].job_id)
        if not relevant:
            continue
        seen = set(past[past.user_id == u.user_id].job_id)
        for method in results:
            idx, _ = rec.recommend(u.resume, u.level, u.city, K, exclude=seen, method=method)
            ids = list(rec.jobs.job_id.values[idx])
            results[method]["P@10"].append(precision_at_k(ids, relevant))
            results[method]["R@10"].append(recall_at_k(ids, relevant))
            results[method]["NDCG@10"].append(ndcg_at_k(ids, relevant))

    lines = [f"Embedder: {rec.emb.name}  |  users evaluated: {len(results['ranked']['P@10'])}", "",
             "| method | P@10 | R@10 | NDCG@10 |", "|---|---|---|---|"]
    for m, r in results.items():
        lines.append(f"| {m} | {np.mean(r['P@10']):.3f} | {np.mean(r['R@10']):.3f} | {np.mean(r['NDCG@10']):.3f} |")
    out = "\n".join(lines)
    print(out)
    open("results.md", "w").write(out + "\n")


if __name__ == "__main__":
    main()
