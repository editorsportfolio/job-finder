"""Makes fake (but believable) jobs, users and clicks so the project runs anywhere."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# skills grouped by career track
TRACKS = {
    "ml":       ["python", "pytorch", "scikit-learn", "pandas", "nlp", "embeddings", "llm", "statistics"],
    "backend":  ["python", "fastapi", "sql", "postgresql", "docker", "redis", "rest", "aws"],
    "frontend": ["javascript", "react", "typescript", "css", "html", "figma", "testing", "nextjs"],
    "data":     ["sql", "excel", "tableau", "python", "pandas", "statistics", "powerbi", "etl"],
    "devops":   ["docker", "kubernetes", "aws", "terraform", "linux", "cicd", "monitoring", "bash"],
}
ALL_SKILLS = sorted({s for pool in TRACKS.values() for s in pool})
LEVELS = ["intern", "junior", "mid", "senior"]
CITIES = ["bangalore", "delhi", "mumbai", "remote"]


def make_jobs(n=2000):
    rows = []
    for i in range(n):
        track = rng.choice(list(TRACKS))
        skills = list(rng.choice(TRACKS[track], size=rng.integers(4, 7), replace=False))
        level = rng.choice(LEVELS, p=[0.25, 0.35, 0.25, 0.15])
        rows.append({
            "job_id": i,
            "title": f"{level} {track} engineer",
            "skills": " ".join(skills),
            "description": f"{level} {track} role. We need {', '.join(skills)}.",
            "level": level,
            "city": rng.choice(CITIES),
            "days_old": int(rng.integers(0, 60)),
            "hype": rng.lognormal(0, 0.8),  # some jobs are just more popular
        })
    return pd.DataFrame(rows)


def make_users(n=300):
    rows = []
    for i in range(n):
        track = rng.choice(list(TRACKS))
        skills = list(rng.choice(TRACKS[track], size=rng.integers(4, 7), replace=False))
        rows.append({
            "user_id": i,
            "skills": " ".join(skills),
            "resume": f"Student interested in {track}. Projects using {', '.join(skills)}.",
            "level": rng.choice(["intern", "junior"], p=[0.5, 0.5]),
            "city": rng.choice(CITIES),
        })
    return pd.DataFrame(rows)


def make_clicks(users, jobs, per_user=60):
    """label: 1 = viewed, 2 = clicked, 3 = saved. Better-matching jobs get more clicks."""
    rows = []
    job_skills = [set(s.split()) for s in jobs.skills]
    for _, u in users.iterrows():
        mine = set(u.skills.split())
        fit = np.array([len(mine & js) / len(js) for js in job_skills]) * 3
        fit += (jobs.level.values == u.level) * 1.0 + (jobs.city.values == u.city) * 0.5
        fit += np.log(jobs.hype.values)
        p = np.exp(fit) / np.exp(fit).sum()
        picked = rng.choice(len(jobs), size=per_user, replace=False, p=p)
        for j in picked:
            chance = 1 / (1 + np.exp(-2 * (fit[j] - 2.5)))
            r = rng.random()
            label = 3 if r < chance * 0.3 else 2 if r < chance else 1
            rows.append({"user_id": u.user_id, "job_id": j, "label": label,
                         "day": int(rng.integers(0, 60))})
    return pd.DataFrame(rows)
