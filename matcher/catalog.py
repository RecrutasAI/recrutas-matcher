"""The deduplicated catalog count Recrutas shows on its homepage: every open US job posted directly by
its employer that users are matched against, counted once (Recrutas's own jobs + Fleet jobs).

The rules come from Recrutas in the inbox (third-party companies and sources, the unnamed-employer
pattern), so both sides count the same way. Links are compared after the Fleet's normalisation.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Same as jobs-fleet fleet/ids.py canonical_url (ADR 0002).
TRACKING_PARAMS = frozenset({"gh_src", "source", "src", "ref", "referrer", "lever-source", "lever-origin", "s", "jobboard", "_ga"})


def canonical_url(url: str | None) -> str:
    if not url:
        return ""
    parts = urlsplit(str(url).strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(sorted(query)), ""))


def link_key(url: str | None) -> str:
    """Same as Recrutas's server/lib/canonical-url.ts linkKey: 16 hex chars of SHA-1 of the canonical link."""
    return hashlib.sha1(canonical_url(url).encode("utf-8")).hexdigest()[:16]


def catalog(state: Path, recrutas: dict) -> dict:
    third = {n.strip().lower() for n in recrutas.get("third_party_companies", [])}
    unnamed = re.compile(recrutas.get("unnamed_employer") or r"(?!)", re.IGNORECASE)
    rec_links = set(recrutas.get("link_keys", []))
    companies = {c.strip().lower() for c in recrutas.get("companies", [])}
    fleet_links = set()
    # Open jobs in every country, same rules but no US filter, for the "we track N open jobs, M of
    # them in the US" line. Recrutas only shows US jobs; this is the Fleet's whole open catalog.
    all_links = set()
    f = pq.ParquetFile(state)
    cols = [c for c in ("url", "company_name", "closed_at", "is_us", "duplicate_of") if c in f.schema_arrow.names]
    for batch in f.iter_batches(batch_size=100_000, columns=cols):
        t = pa.Table.from_batches([batch])
        open_ = pc.is_null(t.column("closed_at"))
        if "duplicate_of" in t.column_names:  # the Fleet marks a job repeated by a second board (2026-10-11)
            open_ = pc.and_(open_, pc.is_null(t.column("duplicate_of")))
        t = t.filter(open_)
        us = pc.fill_null(pc.equal(t.column("is_us"), True), False).to_pylist()
        for url, co, is_us in zip(t.column("url").to_pylist(), t.column("company_name").to_pylist(), us):
            name = (co or "").strip()
            if not name or name.lower() in third or unnamed.search(name):
                continue
            key = link_key(url)
            all_links.add(key)
            if is_us:
                fleet_links.add(key)
                companies.add(name.lower())
    empty = link_key("")
    rec_links.discard(empty)
    fleet_links.discard(empty)
    all_links.discard(empty)
    companies.discard("")
    return {"jobs": len(rec_links | fleet_links), "companies": len(companies), "recrutas": len(rec_links),
            "fleet": len(fleet_links), "both": len(rec_links & fleet_links), "all_countries": len(rec_links | all_links)}
