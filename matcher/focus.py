"""Jobs for Recrutas's public focus pages (/jobs/pierce-care): every open US job in a set of towns.

Recrutas sends each page's towns in the links document ({"focus": [{"slug", "places": [[city, state]]}]})
and its rules for who isn't a real employer (third-party companies, the unnamed-employer pattern), so
both sides filter the same way. Recrutas picks and orders what to show (job types, pay, staffing
firms); this only selects by place and freshness and returns the fields a page shows, plus counts.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

COLUMNS = ["url", "company_name", "title", "location", "city", "us_state", "salary_summary", "salary_min",
           "salary_max", "salary_currency", "salary_period", "posted_at", "last_seen", "closed_at", "is_us",
           "duplicate_of", "is_remote", "description"]
MAX_JOBS = 3000  # per page; a county has about a thousand


def _when(v) -> datetime | None:
    if not v:
        return None
    try:
        d = datetime.fromisoformat(str(v))  # Python 3.11+ reads a trailing Z
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def _pay(r: dict) -> str | None:
    """The posting's own pay words when it has them, else its range as stated (USD only)."""
    s = (r.get("salary_summary") or "").strip()
    if s:
        return s
    lo, hi = r.get("salary_min"), r.get("salary_max")
    if not lo and not hi:
        return None
    if (r.get("salary_currency") or "USD").upper() != "USD":
        return None
    per = {"hour": " an hour", "hourly": " an hour", "year": " a year", "yearly": " a year", "annual": " a year",
           "month": " a month", "week": " a week", "day": " a day"}.get((r.get("salary_period") or "").lower(), "")
    money = lambda v: f"${v:,.2f}".replace(".00", "") if v < 1000 else f"${v:,.0f}"
    if lo and hi and lo != hi:
        return f"{money(lo)} to {money(hi)}{per}"
    return f"{money(lo or hi)}{per}"


def _pay_hint(description: str | None) -> str | None:
    """The bits of the description that start with "$" (at most 5, 70 characters each): Recrutas reads
    pay from them with its own rules (server/lib/job-batch.ts payFromText), so the rules live in one place."""
    if not description:
        return None
    parts = [description[m.start():m.start() + 70].replace("\n", " ") for m in re.finditer(r"\$\s?\d", description)][:5]
    return " | ".join(parts) or None


def focus_jobs(state: Path, recrutas: dict, now: datetime | None = None) -> dict:
    """{slug: {"live", "new_week", "jobs": [...]}} for each focus page in the links document."""
    pages = recrutas.get("focus") or []
    if not pages:
        return {}
    now = now or datetime.now(UTC)
    fresh_after, new_after = now - timedelta(days=3), now - timedelta(days=7)
    third = {n.strip().lower() for n in recrutas.get("third_party_companies", [])}
    unnamed = re.compile(recrutas.get("unnamed_employer") or r"(?!)", re.IGNORECASE)
    wanted = {p["slug"]: {(c.strip().lower(), s.strip().upper()) for c, s in p.get("places", [])} for p in pages}
    any_place = set().union(*wanted.values())
    out = {slug: {"live": 0, "new_week": 0, "jobs": []} for slug in wanted}
    f = pq.ParquetFile(state)
    cols = [c for c in COLUMNS if c in f.schema_arrow.names]
    for batch in f.iter_batches(batch_size=20_000, columns=cols):  # descriptions are large
        t = pa.Table.from_batches([batch])
        keep = pc.and_(pc.is_null(t.column("closed_at")), pc.fill_null(pc.equal(t.column("is_us"), True), False))
        if "duplicate_of" in t.column_names:
            keep = pc.and_(keep, pc.is_null(t.column("duplicate_of")))
        t = t.filter(keep)
        for r in t.to_pylist():
            place = ((r.get("city") or "").strip().lower(), (r.get("us_state") or "").strip().upper())
            if place not in any_place or r.get("is_remote"):
                continue
            name = (r.get("company_name") or "").strip()
            if not name or name.lower() in third or unnamed.search(name) or not r.get("url") or not r.get("title"):
                continue
            seen = _when(r.get("last_seen"))
            if not seen or seen < fresh_after:
                continue  # not seen on the employer's site in the last 3 days
            posted = _when(r.get("posted_at"))
            for slug, places in wanted.items():
                if place not in places:
                    continue
                page = out[slug]
                page["live"] += 1
                if posted and posted > new_after:
                    page["new_week"] += 1
                if len(page["jobs"]) < MAX_JOBS:
                    page["jobs"].append({"title": r["title"].strip(), "company": name, "location": (r.get("location") or "").strip() or None,
                                         "pay": _pay(r), "pay_hint": None if _pay(r) else _pay_hint(r.get("description")),
                                         "url": r["url"], "posted_at": posted.isoformat() if posted else None})
    return out
