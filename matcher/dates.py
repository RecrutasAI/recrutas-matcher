"""Posting dates for Recrutas's jobs, from the Fleet.

The Fleet reads every board in full and keeps the employer's own posting date. Recrutas sends its
live jobs that have no date, by job key ("smartrecruiters:744000154370085", read from the posting
link with the same patterns as Recrutas's server/lib/ats-job-key.ts) and by link fingerprint for
links without a job id. This returns the Fleet's date for each one it holds. Links differ between
the two (a company's own Greenhouse domain, "AccorHotel" vs "accorhotel"), the job id does not.
A key the Fleet holds with different dates (the same id at two companies) is left out.
"""
from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq

from .catalog import link_key

# Same as Recrutas server/lib/ats-job-key.ts VENDOR_ID_PATTERNS.
VENDOR_ID_PATTERNS = {
    "greenhouse": [re.compile(r"[?&]gh_jid=(\d+)"), re.compile(r"greenhouse\.io/[^/]+/jobs/(\d+)")],
    "lever": [re.compile(r"jobs\.lever\.co/[^/]+/([0-9a-f-]{36})", re.IGNORECASE)],
    "ashby": [re.compile(r"jobs\.ashbyhq\.com/[^/]+/([0-9a-f-]{36})", re.IGNORECASE)],
    "smartrecruiters": [re.compile(r"smartrecruiters\.com/[^/]+/(\d+)")],
    "breezy": [re.compile(r"\.breezy\.hr/p/([0-9a-f]{12})", re.IGNORECASE)],
    "workable": [re.compile(r"/j/([0-9A-Za-z]+)")],
    "recruitee": [re.compile(r"recruitee\.com/o/([^/?#]+)")],
}


def job_key(ats: str | None, url: str | None) -> str | None:
    """"smartrecruiters:744000154370085", or None when the link carries no job id."""
    for pattern in VENDOR_ID_PATTERNS.get((ats or "").lower(), []):
        m = pattern.search(url or "")
        if m:
            return f"{ats.lower()}:{m.group(1).lower()}"
    return None


def posting_dates(state: Path, wanted_jobs: set[str], wanted_links: set[str] = frozenset()) -> dict[str, str]:
    """{job key or link fingerprint: the Fleet's posting date} for the ones asked."""
    found: dict[str, set[str]] = {}
    if not wanted_jobs and not wanted_links:
        return {}
    for batch in pq.ParquetFile(state).iter_batches(batch_size=100_000, columns=["ats", "url", "posted_at"]):
        d = batch.to_pydict()
        for ats, url, posted in zip(d["ats"], d["url"], d["posted_at"]):
            if not posted:
                continue
            posted = posted.isoformat() if hasattr(posted, "isoformat") else str(posted)
            jk = job_key(ats, url)
            for key in ((jk,) if jk in wanted_jobs else ()) + ((link_key(url),) if wanted_links and link_key(url) in wanted_links else ()):
                found.setdefault(key, set()).add(posted)
    # One date per key; a job id the Fleet holds with different dates is two different jobs.
    return {k: min(v) for k, v in found.items() if len({x[:10] for x in v}) == 1}
