"""The deduplicated catalog count Recrutas shows on its homepage: every open US job posted directly by
its employer that users are matched against, counted once (Recrutas's own jobs + Fleet jobs).

The rules come from Recrutas in the inbox (third-party companies and sources, the unnamed-employer
pattern), so both sides count the same way. Links are compared after the Fleet's normalisation.
"""
from __future__ import annotations

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


def catalog(state: Path, recrutas: dict) -> dict:
    third = {n.strip().lower() for n in recrutas.get("third_party_companies", [])}
    unnamed = re.compile(recrutas.get("unnamed_employer") or r"(?!)", re.IGNORECASE)
    rec_links, companies = set(), set()
    for url, company in recrutas.get("links", []):
        rec_links.add(canonical_url(url))
        companies.add((company or "").strip().lower())
    fleet_links = set()
    for batch in pq.ParquetFile(state).iter_batches(batch_size=100_000, columns=["url", "company_name", "closed_at", "is_us"]):
        t = pa.Table.from_batches([batch])
        t = t.filter(pc.and_(pc.is_null(t.column("closed_at")), pc.fill_null(pc.equal(t.column("is_us"), True), False)))
        for url, co in zip(t.column("url").to_pylist(), t.column("company_name").to_pylist()):
            name = (co or "").strip()
            if not name or name.lower() in third or unnamed.search(name):
                continue
            fleet_links.add(canonical_url(url))
            companies.add(name.lower())
    rec_links.discard("")
    fleet_links.discard("")
    companies.discard("")
    return {"jobs": len(rec_links | fleet_links), "companies": len(companies), "recrutas": len(rec_links),
            "fleet": len(fleet_links), "both": len(rec_links & fleet_links)}
