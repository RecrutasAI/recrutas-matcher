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
    # Newest posting first.
    assert [j["pay"] for j in out["jobs"]] == ["$24 to $30 an hour", None, "$45 to $60 per hour"]
    assert out["jobs"][0] == {"title": "Behavioral Health Technician", "company": "MultiCare", "location": "Tacoma, WA",
                              "pay": "$24 to $30 an hour", "pay_hint": None, "url": "https://jobs.lever.co/a/1",
                              "posted_at": "2026-11-18T00:00:00+00:00"}
    assert out["jobs"][1]["pay_hint"] == "$22.50 - $28.00 per hour, plus a $500 bonus."


def test_no_focus_pages_no_work(tmp_path: Path):
    assert focus_jobs(tmp_path / "missing.parquet", {}) == {}


def test_pay_only_in_dollars_and_as_stated():
    from matcher.focus import _pay
    base = {"salary_summary": None, "salary_currency": "USD", "salary_period": "hour"}
    assert _pay({**base, "salary_min": 0.0, "salary_max": 50.0}) == "up to $50 an hour"
    assert _pay({**base, "salary_min": 30.0, "salary_max": 24.0}) == "$24 to $30 an hour"
    assert _pay({**base, "salary_summary": "DOE", "salary_min": 24.0, "salary_max": 30.0}) == "$24 to $30 an hour"
    assert _pay({**base, "salary_summary": "C$45 - C$55 an hour", "salary_currency": "CAD"}) is None
    assert _pay({**base, "salary_summary": "CA$45 - CA$55 an hour", "salary_currency": None}) is None
    assert _pay({**base, "salary_min": 24.0, "salary_max": float("nan")}) == "$24 an hour"
    assert _pay({**base, "salary_min": 40.0, "salary_max": 50.0, "salary_currency": "CAD"}) is None
    assert _pay({**base, "salary_min": 40.0, "salary_max": 50.0, "salary_currency": None}) is None
    assert _pay({**base, "salary_summary": "€45,000 - €55,000"}) is None
    assert _pay({**base, "salary_summary": "$45,000 - $55,000 a year"}) == "$45,000 - $55,000 a year"
    assert _pay({**base, "salary_summary": "USD 22–28 / hour"}) == "USD 22–28 / hour"
    assert _pay({**base, "salary_summary": "DOE", "salary_min": None, "salary_max": None}) is None


def test_job_board_links_and_bad_places_are_left_out(tmp_path: Path):
    state = tmp_path / "s.parquet"
    pq.write_table(pa.Table.from_pylist([row(), row(url="https://www.indeed.com/viewjob?jk=1")]), state)
    rec = {"focus": [{"slug": "p", "places": [["tacoma", "WA"], ["oops", None], ["a", "b", "c"]]}, {"slug": "empty", "places": [[None, "WA"]]}]}
    out = focus_jobs(state, rec, now=NOW)
    assert list(out) == ["p"] and out["p"]["live"] == 1
