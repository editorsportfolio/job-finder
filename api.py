"""FastAPI wrapper. Run: uvicorn api:app --reload"""
from fastapi import FastAPI
from pydantic import BaseModel

from data import make_jobs, make_users, make_clicks
from recommender import Recommender, find_skills

app = FastAPI(title="Jobs-for-You")

# builds the data + trains the ranker once on startup (takes a few seconds)
jobs, users = make_jobs(), make_users()
rec = Recommender(jobs, users, make_clicks(users, jobs))


class ColdStartRequest(BaseModel):
    resume_text: str
    level: str = "junior"
    city: str = "remote"
    k: int = 10


class UserRequest(BaseModel):
    user_id: int
    k: int = 10


def format_results(idx, scores, resume):
    mine = find_skills(resume)
    out = []
    for i, s in zip(idx, scores):
        job = rec.jobs.iloc[i]
        out.append({
            "job_id": int(job.job_id),
            "title": job.title,
            "city": job.city,
            "score": round(float(s), 4),
            "matched_skills": sorted(mine & set(job.skills.split())),
        })
    return out


@app.get("/health")
def health():
    return {"status": "ok", "embedder": rec.emb.name}


@app.post("/recommend")
def recommend(req: UserRequest):
    u = users[users.user_id == req.user_id].iloc[0]
    idx, scores = rec.recommend(u.resume, u.level, u.city, req.k)
    return format_results(idx, scores, u.resume)


@app.post("/recommend/cold-start")
def cold_start(req: ColdStartRequest):
    """New user, no history: we only have their resume text."""
    idx, scores = rec.recommend(req.resume_text, req.level, req.city, req.k)
    return format_results(idx, scores, req.resume_text)
