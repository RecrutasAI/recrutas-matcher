from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from matcher.catalog import canonical_url, catalog, link_key


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
    rec = {"link_keys": [link_key("https://jobs.lever.co/acme/1/?utm_source=li"), link_key("https://boards.greenhouse.io/beta/jobs/9")],
           "companies": ["acme", "beta"],
           "third_party_companies": ["collabera"], "unnamed_employer": "saasfaprod|oraclecloud\\\\.com"}
    assert catalog(state, rec) == {"jobs": 3, "companies": 2, "recrutas": 2, "fleet": 2, "both": 1}


def test_link_key_matches_recrutas():
    # Recrutas computes linkKey('https://jobs.lever.co/acme/1') in TypeScript; same bytes here.
    import hashlib
    assert link_key("https://jobs.lever.co/acme/1/?utm_source=li") == hashlib.sha1(b"https://jobs.lever.co/acme/1").hexdigest()[:16]
