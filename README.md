# Jobs-for-You: a small job recommender

Recommends jobs to students in two steps, the way real recommenders do:

1. **Retrieval:** embed the resume and every job, take the 100 closest jobs by cosine similarity.
2. **Ranking:** a LightGBM LambdaRank model re-orders those 100 using skills overlap,
   seniority match, city match, job age and popularity.

New users with no history (**cold start**) work too, since all features come from the resume text.

## Run it

```bash
pip install -r requirements.txt
python evaluate.py                # compares 3 methods, writes results.md
uvicorn api:app --reload          # http://127.0.0.1:8000/docs
```

Docker: `docker build -t jobs-for-you . && docker run -p 8000:8000 jobs-for-you`

No internet for the model download? Run with `EMBEDDER=tfidf` and it uses TF-IDF + SVD instead.

## Results (synthetic data, TF-IDF embedder, time-based split)

| method | P@10 | R@10 | NDCG@10 |
|---|---|---|---|
| popularity | 0.039 | 0.053 | 0.052 |
| embeddings only | 0.036 | 0.048 | 0.049 |
| embeddings + LightGBM ranker | 0.094 | 0.123 | 0.124 |

Run it yourself and use your own numbers. They change with the embedder (MiniLM vs TF-IDF).

## API

- `GET /health`
- `POST /recommend` with `{"user_id": 3, "k": 10}`
- `POST /recommend/cold-start` with `{"resume_text": "...", "level": "junior", "city": "remote"}`

## Files

| file | what it does |
|---|---|
| `data.py` | makes fake jobs, users and clicks |
| `recommender.py` | embeddings, retrieval, features, ranker |
| `evaluate.py` | precision@10, recall@10, NDCG@10 and the comparison |
| `api.py` | FastAPI endpoints |

## How the split works (no leaking)

- days 0-29: used only to count job popularity
- days 30-44: clicks the ranker learns from
- days 45-59: held out for testing; only clicks and saves count as relevant
- jobs a user already saw are never recommended

## Next steps

Swap the in-memory similarity search for Postgres + pgvector, and train on real clicks.
