"""python -m matcher.cli [--dry-run]: read the inbox, match, write the outbox. Logs counts only."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .catalog import catalog
from .dates import posting_dates
from .focus import focus_jobs
from .match import run

JOBS_DATASET = os.environ.get("JOBS_DATASET", "recrutas/us-jobs")
MATCHER_DATASET = os.environ.get("MATCHER_DATASET", "recrutas/matcher")
INBOX = "inbox/candidates.json"
OUTBOX = "outbox/latest.json.gz"
LINKS = "inbox/recrutas-links.json"


def already_matched(inbox: dict, outbox_inbox_exported_at: str | None) -> bool:
    """The outbox was made from this very inbox (a second scheduled slot after a run that did happen)."""
    return bool(inbox.get("exported_at")) and inbox.get("exported_at") == outbox_inbox_exported_at


def main() -> int:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--if-new", action="store_true", help="skip when the outbox already matches this inbox")
    args = p.parse_args()

    inbox = json.loads(Path(hf_hub_download(MATCHER_DATASET, INBOX, repo_type="dataset", force_download=True)).read_text())
    if args.if_new:
        # GitHub drops some scheduled runs, so each cycle has two slots; the second exits here
        # when the first already ran.
        try:
            with gzip.open(hf_hub_download(MATCHER_DATASET, OUTBOX, repo_type="dataset", force_download=True), "rt", encoding="utf-8") as f:
                done = json.load(f).get("inbox_exported_at")
        except EntryNotFoundError:
            done = None
        if already_matched(inbox, done):
            print("outbox already matches this inbox; nothing to do")
            return 0
    vectors = Path(hf_hub_download(JOBS_DATASET, "vectors/jobs.parquet", repo_type="dataset"))
    state = Path(hf_hub_download(JOBS_DATASET, "state/jobs.parquet", repo_type="dataset"))
    result = run(vectors, state, inbox)
    # The homepage's deduplicated count, when the server sent its own direct links.
    dates: dict[str, str] = {}
    focus: dict | None = None
    links_doc: dict = {}
    try:
        links_doc = json.loads(Path(hf_hub_download(MATCHER_DATASET, LINKS, repo_type="dataset", force_download=True)).read_text(encoding="utf-8"))
        result["stats"]["catalog"] = catalog(state, links_doc)
        # The Fleet's posting dates for Recrutas's undated jobs.
        asked_jobs, asked_links = set(links_doc.get("undated_job_keys", [])), set(links_doc.get("undated_link_keys", []))
        dates = posting_dates(state, asked_jobs, asked_links)
        result["stats"]["dates"] = {"asked": len(asked_jobs) + len(asked_links), "found": len(dates)}
    except Exception as e:  # noqa: BLE001 -- never block matching on the count or the dates
        print(f"catalog count / dates skipped: {type(e).__name__}")
    try:
        # Every open job in the towns of Recrutas's public focus pages (/jobs/pierce-care).
        focus = focus_jobs(state, links_doc) if links_doc else None
        if focus is not None:
            result["stats"]["focus"] = {slug: {"live": v["live"], "listed": len(v["jobs"])} for slug, v in focus.items()}
    except Exception as e:  # noqa: BLE001 -- never block matching on the focus pages
        print(f"focus pages skipped: {type(e).__name__}")
    print(json.dumps(result["stats"]))
    out = Path(tempfile.mkdtemp()) / "latest.json.gz"
    payload = {"generated_at": datetime.now(UTC).isoformat(), "inbox_exported_at": inbox.get("exported_at"),
               "jobs": result["jobs"], "closed": result["closed"], "closed_detail": result["closed_detail"],
               "seen": result["seen"], "dates": dates, "stats": result["stats"]}
    if focus is not None:  # left out when it failed: Recrutas keeps each page's last good list
        payload["focus"] = focus
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"outbox: {out.stat().st_size // 1024} KB")
    if not args.dry_run:
        HfApi().upload_file(path_or_fileobj=str(out), path_in_repo=OUTBOX, repo_id=MATCHER_DATASET, repo_type="dataset",
                            commit_message=f"Matches: {result['stats']['sent']} jobs for {result['stats']['users']} users")
        print("outbox committed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
