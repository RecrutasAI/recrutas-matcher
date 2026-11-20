from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from matcher.focus import focus_jobs

NOW = datetime(2026, 11, 20, 12, tzinfo=UTC)


def row(**o):
    r = {"url": "https://jobs.lever.co/a/1", "company_name": "MultiCare", "title": "Behavioral Health Technician",
         "location": "Tacoma, WA", "city": "Tacoma", "us_state": "WA", "salary_summary": None, "salary_min": 24.0,
         "salary_max": 30.0, "salary_currency": "USD", "salary_period": "hour", "posted_at": "2026-11-18T00:00:00Z", "description": None,
         "last_seen": "2026-11-20T06:00:00+00:00", "closed_at": None, "is_us": True, "duplicate_of": None, "is_remote": False}
    r.update(o)
    return r


def test_open_fresh_jobs_in_the_towns_with_pay_and_counts(tmp_path: Path):
    state = tmp_path / "s.parquet"
    pq.write_table(pa.Table.from_pylist([
        row(),
        row(url="https://x/2", title="RN", salary_min=None, salary_max=None, salary_summary="$45 to $60 per hour", posted_at="2026-10-01T00:00:00Z"),
        row(url="https://x/3", city="Seattle"),                                  # another town
        row(url="https://x/4", closed_at="2026-11-19"),                           # closed
        row(url="https://x/5", last_seen="2026-11-10T00:00:00Z"),                 # not seen in 3 days
        row(url="https://x/6", company_name="Collabera"),                         # third party
        row(url="https://x/7", is_remote=True),                                   # remote
        row(url="https://x/8", duplicate_of="x"),                                 # duplicate
        row(url="https://x/9", salary_min=None, salary_max=None,                  # pay only in the text: a hint
            description="Great team.\nPay: $22.50 - $28.00 per hour, plus a $500 bonus."),
    ]), state)
    rec = {"focus": [{"slug": "pierce-care", "places": [["tacoma", "WA"], ["lakewood", "WA"]]}],
           "third_party_companies": ["collabera"], "unnamed_employer": "saasfaprod"}
    out = focus_jobs(state, rec, now=NOW)["pierce-care"]
    assert out["live"] == 3 and out["new_week"] == 2
    assert [j["pay"] for j in out["jobs"]] == ["$24 to $30 an hour", "$45 to $60 per hour", None]
    assert out["jobs"][0] == {"title": "Behavioral Health Technician", "company": "MultiCare", "location": "Tacoma, WA",
                              "pay": "$24 to $30 an hour", "pay_hint": None, "url": "https://jobs.lever.co/a/1",
                              "posted_at": "2026-11-18T00:00:00+00:00"}
    assert out["jobs"][2]["pay_hint"] == "$22.50 - $28.00 per hour, plus a $500 bonus. | $28.00 per hour, plus a $500 bonus. | $500 bonus."


def test_no_focus_pages_no_work(tmp_path: Path):
    assert focus_jobs(tmp_path / "missing.parquet", {}) == {}
