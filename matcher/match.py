"""Match Recrutas users to Fleet jobs on GitHub's machines, so the Recrutas server doesn't have to.

Inputs (private storage): each user's resume vector under an anonymous id, the Fleet jobs Recrutas
already shows, the Fleet's job vectors and current job state. Output: every job in any user's top
matches, ready to insert (details + vector), and which shown jobs have closed. The server only
inserts and closes. Recrutas's feed still applies all its own rules to whatever arrives.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TOP_K = 200
JOB_COLUMNS = ["job_id", "title", "company_name", "location", "description", "url", "apply_url", "ats", "posted_at",
               "is_remote", "salary_min", "salary_max", "salary_currency", "closed_at", "is_us", "last_seen"]

# Staffing agencies are kept: Recrutas labels them third-party and shows them only when a person has
# no direct matches (server/lib/third-party.ts). Links that point at a job board are dropped here,
# since they would carry an ATS source into Recrutas.
# Read on the link's HOST only, whole labels: "?source=linkedin.com" in a Greenhouse link and hosts like
# "acmetalent.com" are not job boards. Google's own careers pages (google.com/about/careers, and
# GFiber's fiber.google.com/careers) are the employer; any other google.com link (Google for Jobs)
# is a board. Measured 2026-11-21: the old rule dropped 1,788 open US jobs, all Google and GFiber.
JOB_BOARD_HOSTS = re.compile(r"(^|\.)((linkedin|indeed|glassdoor|ziprecruiter|monster|simplyhired|dice|careerbuilder|wellfound|"
                             r"builtin|jooble|adzuna|talent|jobgether|google|hiringcafe|otta)\.(com|org)|hiring\.cafe)$", re.IGNORECASE)
GOOGLE_HOST = re.compile(r"(^|\.)google\.com$", re.IGNORECASE)


def is_job_board(link: str) -> bool:
    parts = urlsplit(link)
    host = (parts.hostname or "").lower()
    if GOOGLE_HOST.search(host) and "/careers/" in parts.path:
        return False
    return bool(JOB_BOARD_HOSTS.search(host))


def load_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    t = pq.read_table(path, columns=["job_id", "vector"])
    ids = t.column("job_id").to_pylist()
    flat = pc.list_flatten(t.column("vector")).to_numpy(zero_copy_only=False).astype(np.float32)
    v = flat.reshape(len(ids), -1)
    return ids, v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def candidate_matrix(candidates: list[dict], dim: int) -> tuple[list[str], np.ndarray]:
    ok = [c for c in candidates if isinstance(c.get("v"), list) and len(c["v"]) == dim]
    m = np.array([c["v"] for c in ok], dtype=np.float32).reshape(len(ok), dim)
    return [c["pid"] for c in ok], m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def top_k(users: np.ndarray, jobs: np.ndarray, allowed: np.ndarray, k: int = TOP_K, chunk: int = 256) -> list[list[tuple[int, float]]]:
    """Each user's k nearest allowed jobs (index, cosine), best first."""
    out: list[list[tuple[int, float]]] = []
    k = min(k, int(allowed.sum()))
    if k == 0:
        return [[] for _ in range(len(users))]
    for i in range(0, len(users), chunk):
        s = users[i: i + chunk] @ jobs.T
        s[:, ~allowed] = -np.inf
        idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
        for row, cols in zip(s, idx):
            best = cols[np.argsort(-row[cols])]
            out.append([(int(j), float(row[j])) for j in best])
    return out


def job_rows(state: Path, wanted: set[str]) -> dict[str, dict]:
    """Current state rows for the wanted job ids (streamed: the state file is large)."""
    rows: dict[str, dict] = {}
    f = pq.ParquetFile(state)
    cols = [c for c in JOB_COLUMNS + ["duplicate_of"] if c in f.schema_arrow.names]
    for batch in f.iter_batches(batch_size=50_000, columns=cols):
        t = pa.Table.from_batches([batch])
        hit = t.filter(pc.is_in(t.column("job_id"), value_set=pa.array(list(wanted), pa.string())))
        for r in hit.to_pylist():
            rows[r["job_id"]] = r
    return rows


def first_cut(row: dict) -> bool:
    """Open, US, one copy of the job (not marked duplicate_of), a link that isn't a job board's."""
    if row.get("closed_at") is not None or row.get("is_us") is not True or row.get("duplicate_of"):
        return False
    link = row.get("url") or row.get("apply_url") or ""
    return link.startswith("http") and not is_job_board(link)


def iso(v) -> str | None:
    return v.isoformat() if hasattr(v, "isoformat") else v


def closure(job_id: str, row: dict | None) -> dict:
    """Why a shown job closed. Only "taken_down" is a takedown: the Fleet sets closed_at when the job
    is missing from a successful read of its board. A job gone from the catalog or no longer US closed
    for other reasons, and Recrutas mustn't tell anyone it was taken down."""
    if row is None:
        return {"id": job_id, "reason": "missing"}
    if row.get("closed_at") is not None:
        return {"id": job_id, "reason": "taken_down", "at": iso(row["closed_at"])}
    return {"id": job_id, "reason": "not_us"}


def run(vectors: Path, state: Path, inbox: dict, k: int = TOP_K) -> dict:
    ids, jobs = load_vectors(vectors)
    pids, users = candidate_matrix(inbox.get("candidates", []), jobs.shape[1])
    allowed = np.ones(len(ids), dtype=bool)
    picks = top_k(users, jobs, allowed, k) if len(pids) else []
    wanted = {ids[j] for p in picks for j, _ in p}
    held = set(inbox.get("held", []))
    rows = job_rows(state, wanted | held)
    index = {jid: i for i, jid in enumerate(ids)}
    out_jobs = []
    for jid in sorted(wanted):
        r = rows.get(jid)
        if r is None or not first_cut(r):
            continue
        r = {key: (val.isoformat() if hasattr(val, "isoformat") else val) for key, val in r.items() if key not in ("closed_at", "is_us", "duplicate_of", "last_seen")}
        r["vector"] = [round(float(x), 5) for x in jobs[index[jid]]]
        out_jobs.append(r)
    closed = sorted(j for j in held if j not in rows or rows[j].get("closed_at") is not None or rows[j].get("is_us") is not True)
    return {"jobs": out_jobs, "closed": closed, "closed_detail": [closure(j, rows.get(j)) for j in closed],
            "seen": {j: iso(rows[j].get("last_seen")) for j in sorted(held - set(closed)) if iso(rows[j].get("last_seen"))},
            "stats": {"users": len(pids), "fleet_jobs": len(ids), "picked": len(wanted), "sent": len(out_jobs),
                      "held": len(held), "closed": len(closed)}}
