from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from matcher.catalog import link_key
from matcher.dates import job_key, posting_dates


def test_job_key_matches_recrutas():
    assert job_key("smartrecruiters", "https://jobs.smartrecruiters.com/AccorHotel/744000154370085") == "smartrecruiters:744000154370085"
    assert job_key("greenhouse", "https://stripe.com/jobs/search?gh_jid=8172487") == "greenhouse:8172487"
    assert job_key("greenhouse", "https://job-boards.greenhouse.io/stripe/jobs/8172487") == "greenhouse:8172487"
    assert job_key("lever", "https://jobs.lever.co/acme/400AA178-7d5d-4b28-9d72-43a8b6054c11") == "lever:400aa178-7d5d-4b28-9d72-43a8b6054c11"
    assert job_key("jazzhr", "https://x.applytojob.com/apply/1") is None


def test_dates_by_job_id_and_by_link(tmp_path: Path):
    state = tmp_path / "s.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"ats": "smartrecruiters", "url": "https://jobs.smartrecruiters.com/AccorHotel/7", "posted_at": "2026-10-08T08:28:25+00:00"},
        {"ats": "greenhouse", "url": "https://job-boards.greenhouse.io/a/jobs/9", "posted_at": "2026-10-01T00:00:00+00:00"},
        {"ats": "greenhouse", "url": "https://job-boards.greenhouse.io/b/jobs/9", "posted_at": "2026-08-01T00:00:00+00:00"},  # same id, another company
        {"ats": "jazzhr", "url": "https://x.applytojob.com/apply/1", "posted_at": "2026-10-02T00:00:00+00:00"},
        {"ats": "lever", "url": "https://jobs.lever.co/acme/400aa178-7d5d-4b28-9d72-43a8b6054c11", "posted_at": None},
    ]), state)
    got = posting_dates(state, {"smartrecruiters:7", "greenhouse:9", "lever:400aa178-7d5d-4b28-9d72-43a8b6054c11"},
                        {link_key("https://x.applytojob.com/apply/1")})
    assert got == {"smartrecruiters:7": "2026-10-08T08:28:25+00:00", link_key("https://x.applytojob.com/apply/1"): "2026-10-02T00:00:00+00:00"}
