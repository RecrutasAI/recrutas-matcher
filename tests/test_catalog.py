from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from matcher.catalog import canonical_url, catalog


def test_canonical_url_matches_the_fleet():
    assert canonical_url("HTTPS://Jobs.Lever.co/acme/1/?utm_source=x&gh_src=y&b=2&a=1") == "https://jobs.lever.co/acme/1?a=1&b=2"


def test_counts_each_direct_us_job_once(tmp_path: Path):
    state = tmp_path / "s.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"url": "https://jobs.lever.co/acme/1", "company_name": "Acme", "closed_at": None, "is_us": True},     # also in Recrutas
        {"url": "https://jobs.lever.co/acme/2", "company_name": "Acme", "closed_at": None, "is_us": True},
        {"url": "https://jobs.lever.co/acme/3", "company_name": "Acme", "closed_at": "2026-10-01", "is_us": True},  # closed
        {"url": "https://jobs.lever.co/acme/4", "company_name": "Acme", "closed_at": None, "is_us": False},   # not US
        {"url": "https://x.com/jobs/5", "company_name": "Collabera", "closed_at": None, "is_us": True},       # third party
        {"url": "https://x.com/jobs/6", "company_name": "Fa Ertb Saasfaprod1", "closed_at": None, "is_us": True},  # unnamed
    ]), state)
    rec = {"links": [["https://jobs.lever.co/acme/1/?utm_source=li", "acme"], ["https://boards.greenhouse.io/beta/jobs/9", "Beta"]],
           "third_party_companies": ["collabera"], "unnamed_employer": "saasfaprod|oraclecloud\\\\.com"}
    assert catalog(state, rec) == {"jobs": 3, "companies": 2, "recrutas": 2, "fleet": 2, "both": 1}
