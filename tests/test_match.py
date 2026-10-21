from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from matcher import match as M

DIM = 8


def write_vectors(path: Path, vecs: dict[str, list[float]]) -> Path:
    ids = list(vecs)
    flat = pa.array(np.array([vecs[i] for i in ids], dtype=np.float16).reshape(-1), type=pa.float16())
    pq.write_table(pa.table({"job_id": ids, "content_hash": ["h"] * len(ids),
                             "vector": pa.FixedSizeListArray.from_arrays(flat, DIM)}), path)
    return path


def write_state(path: Path, rows: list[dict]) -> Path:
    base = {c: None for c in M.JOB_COLUMNS}
    pq.write_table(pa.Table.from_pylist([{**base, **r} for r in rows]), path)
    return path


def unit(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def test_top_k_orders_best_first_and_respects_allowed():
    jobs = np.eye(4, dtype=np.float32)
    users = np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
    assert [j for j, _ in M.top_k(users, jobs, np.array([True, True, True, True]), k=2)[0]] == [0, 1]
    assert next(j for j, _ in M.top_k(users, jobs, np.array([False, True, True, True]), k=2)[0]) == 1


def test_run_sends_matches_with_vectors_and_reports_closed(tmp_path: Path):
    vectors = write_vectors(tmp_path / "v.parquet", {"j1": unit(0), "j2": unit(1), "j3": unit(2), "j4": unit(3)})
    state = write_state(tmp_path / "s.parquet", [
        {"job_id": "j1", "title": "Nurse", "company_name": "Kaiser", "url": "https://jobs.lever.co/kaiser/1", "is_us": True},
        {"job_id": "j2", "title": "Rep", "company_name": "Jobgether", "url": "https://jobs.lever.co/jobgether/2", "is_us": True},
        {"job_id": "j3", "title": "Chef", "company_name": "Acme", "url": "https://www.linkedin.com/jobs/3", "is_us": True},
        {"job_id": "j4", "title": "Driver", "company_name": "Acme", "url": "https://acme.com/jobs/4", "is_us": True},
        {"job_id": "old1", "title": "x", "company_name": "Acme", "url": "https://acme.com/jobs/9", "is_us": True,
         "closed_at": "2026-10-10T00:00:00+00:00"},
    ])
    inbox = {"candidates": [{"pid": "p1", "v": unit(0)}, {"pid": "bad", "v": [1.0]}], "held": ["old1", "gone", "j4"]}
    out = M.run(vectors, state, inbox, k=4)
    sent = {j["job_id"]: j for j in out["jobs"]}
    assert set(sent) == {"j1", "j2", "j4"}                        # agency kept (labeled by Recrutas); job-board link dropped
    assert len(sent["j1"]["vector"]) == DIM and "closed_at" not in sent["j1"]
    assert out["closed"] == ["gone", "old1"]                      # held jobs now closed or missing
    assert out["stats"]["users"] == 1                              # a malformed vector is ignored


def test_run_says_why_each_shown_job_closed_and_when_open_ones_were_seen(tmp_path: Path):
    vectors = write_vectors(tmp_path / "v.parquet", {"j1": unit(0)})
    state = write_state(tmp_path / "s.parquet", [
        {"job_id": "open1", "title": "a", "company_name": "Acme", "url": "https://acme.com/1", "is_us": True,
         "last_seen": "2026-10-20T08:00:00+00:00"},
        {"job_id": "down1", "title": "b", "company_name": "Acme", "url": "https://acme.com/2", "is_us": True,
         "closed_at": "2026-10-19T09:00:00+00:00", "last_seen": "2026-10-18T09:00:00+00:00"},
        {"job_id": "abroad", "title": "c", "company_name": "Acme", "url": "https://acme.com/3", "is_us": False},
    ])
    out = M.run(vectors, state, {"candidates": [], "held": ["open1", "down1", "abroad", "gone"]}, k=1)
    reasons = {c["id"]: c for c in out["closed_detail"]}
    # Only a job missing from a successful board read is a takedown, with its real time.
    assert reasons["down1"] == {"id": "down1", "reason": "taken_down", "at": "2026-10-19T09:00:00+00:00"}
    assert reasons["abroad"]["reason"] == "not_us"
    assert reasons["gone"]["reason"] == "missing"
    assert out["seen"] == {"open1": "2026-10-20T08:00:00+00:00"}


def test_second_slot_skips_only_an_already_matched_inbox():
    from matcher.cli import already_matched
    assert already_matched({"exported_at": "2026-10-11T17:50:06Z"}, "2026-10-11T17:50:06Z")
    assert not already_matched({"exported_at": "2026-10-11T21:50:06Z"}, "2026-10-11T17:50:06Z")
    assert not already_matched({}, None)
