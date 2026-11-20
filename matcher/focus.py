"""Jobs for Recrutas's public focus pages (/jobs/pierce-care): every open US job in a set of towns.

Recrutas sends each page's towns in the links document ({"focus": [{"slug", "places": [[city, state]]}]})
and its rules for who isn't a real employer (third-party companies, the unnamed-employer pattern), so
both sides filter the same way. Recrutas picks and orders what to show (job types, pay, staffing
firms; third-party companies are left out here, as in the homepage count); this only selects by place and freshness and returns the fields a page shows, plus counts.
Places are matched on the Fleet's city and state columns, so `live` is a floor: a job whose city
column is empty or names another of its locations isn't counted.
"""
from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .match import first_cut

COLUMNS = ["url", "apply_url", "company_name", "title", "location", "city", "us_state", "salary_summary", "salary_min",
           "salary_max", "salary_currency", "salary_period", "posted_at", "last_seen", "closed_at", "is_us",
           "duplicate_of", "is_remote", "description"]
MAX_JOBS = 3000  # per page, newest first; a county has about a thousand
PERIODS = {"hour": " an hour", "hourly": " an hour", "year": " a year", "yearly": " a year", "annual": " a year",
           "month": " a month", "week": " a week", "day": " a day"}


def _when(v) -> datetime | None:
    if not v:
        return None
    try:
        d = datetime.fromisoformat(str(v))  # Python 3.11+ reads a trailing Z
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def _money(v: float) -> str:
    return f"${v:,.2f}".replace(".00", "") if v < 1000 else f"${v:,.0f}"


FOREIGN = re.compile(r"[€£¥₹]|\b(?:CA|AU|C|A|MX|NZ|HK|S|R)\s?\$|\b(CAD|EUR|GBP|AUD|MXN|INR|NZD|HKD|SGD)\b", re.IGNORECASE)


def _pay(r: dict) -> str | None:
    """Pay as the posting states it, in US dollars only: its own words when they are dollars
    ("$22 to $28 an hour", "USD 22–28 / hour"), else its structured US range. A 0 is unstated
    ("up to $30"), never "$0"."""
    s = (r.get("salary_summary") or "").strip()
    currency = (r.get("salary_currency") or "").upper()
    if s and ("$" in s or re.search(r"\bUSD\b", s, re.IGNORECASE)) and not FOREIGN.search(s) and currency in ("", "USD"):
        return s
    if currency != "USD":
        return None
    lo, hi = (v if isinstance(v, (int, float)) and not math.isnan(v) and v > 0 else None for v in (r.get("salary_min"), r.get("salary_max")))  # 0 or NaN: not stated
    if lo is None and hi is None:
        return None
    per = PERIODS.get((r.get("salary_period") or "").lower(), "")
    if lo is not None and hi is not None:
        lo, hi = min(lo, hi), max(lo, hi)
        return f"{_money(lo)}{per}" if lo == hi else f"{_money(lo)} to {_money(hi)}{per}"
    return f"up to {_money(hi)}{per}" if lo is None else f"{_money(lo)}{per}"


def _pay_hint(description: str | None) -> str | None:
    """The bits of the description that start with "$" (at most 5, 70 characters each, not overlapping):
    Recrutas reads pay from them with its own rules (server/lib/job-batch.ts payFromText), so the rules
    live in one place."""
    if not description:
        return None
    parts, end = [], -1
    for m in re.finditer(r"\$\s?\d", description):
        if m.start() < end:
            continue
        parts.append(description[m.start():m.start() + 70].replace("\n", " "))
        end = m.start() + 70
        if len(parts) == 5:
            break
    return " | ".join(parts) or None


def _places(page: dict) -> set[tuple[str, str]]:
    out = set()
    for p in page.get("places") or []:
        if isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(x, str) and x.strip() for x in p):
            out.add((p[0].strip().lower(), p[1].strip().upper()))
    return out


def focus_jobs(state: Path, recrutas: dict, now: datetime | None = None) -> dict:
    """{slug: {"live", "new_week", "jobs": [...]}} for each focus page in the links document."""
    wanted = {p["slug"]: _places(p) for p in recrutas.get("focus") or [] if isinstance(p, dict) and p.get("slug")}
    wanted = {slug: places for slug, places in wanted.items() if places}
    if not wanted:
        return {}
    now = now or datetime.now(UTC)
    fresh_after, new_after = now - timedelta(days=3), now - timedelta(days=7)
    third = {n.strip().lower() for n in recrutas.get("third_party_companies") or [] if isinstance(n, str)}
    unnamed = re.compile(recrutas.get("unnamed_employer") or r"(?!)", re.IGNORECASE)
    any_place = set().union(*wanted.values())
    cities = pa.array(sorted({c for c, _ in any_place}))
    states = pa.array(sorted({s for _, s in any_place}))
    out = {slug: {"live": 0, "new_week": 0, "jobs": []} for slug in wanted}
    f = pq.ParquetFile(state)
    cols = [c for c in COLUMNS if c in f.schema_arrow.names]
    if not {"city", "us_state", "url", "title", "company_name", "last_seen"} <= set(cols):
        raise ValueError(f"state is missing columns focus pages need: {sorted({'city', 'us_state', 'url', 'title', 'company_name', 'last_seen'} - set(cols))}")
    for batch in f.iter_batches(batch_size=50_000, columns=cols):
        t = pa.Table.from_batches([batch])
        # The towns first, in Arrow: only those rows (and their descriptions) become Python objects.
        here = pc.and_(pc.fill_null(pc.is_in(pc.utf8_lower(pc.utf8_trim_whitespace(t.column("city"))), value_set=cities), False),
                       pc.fill_null(pc.is_in(pc.utf8_upper(pc.utf8_trim_whitespace(t.column("us_state"))), value_set=states), False))
        t = t.filter(here)
        for r in t.to_pylist():
            if not first_cut(r) or r.get("is_remote"):
                continue  # open, US, one copy, an employer's own link (as the feed); not remote
            place = ((r.get("city") or "").strip().lower(), (r.get("us_state") or "").strip().upper())
            name = (r.get("company_name") or "").strip()
            if not name or name.lower() in third or unnamed.search(name) or not (r.get("title") or "").strip():
                continue
            seen = _when(r.get("last_seen"))
            if not seen or seen < fresh_after:
                continue  # not seen on the employer's site in the last 3 days
            posted = _when(r.get("posted_at"))
            pay = _pay(r)
            job = {"title": r["title"].strip(), "company": name, "location": (r.get("location") or "").strip() or None,
                   "pay": pay, "pay_hint": None if pay else _pay_hint(r.get("description")),
                   "url": r.get("url") or r.get("apply_url"), "posted_at": posted.astimezone(UTC).isoformat() if posted else None}
            for slug, places in wanted.items():
                if place in places:
                    page = out[slug]
                    page["live"] += 1
                    if posted and posted > new_after:
                        page["new_week"] += 1
                    page["jobs"].append(job)
    for page in out.values():
        page["jobs"].sort(key=lambda j: j["posted_at"] or "", reverse=True)
        del page["jobs"][MAX_JOBS:]
    return out
