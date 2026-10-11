"""python -m matcher.cli [--dry-run]: read the inbox, match, write the outbox. Logs counts only."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .match import run

JOBS_DATASET = os.environ.get("JOBS_DATASET", "recrutas/us-jobs")
MATCHER_DATASET = os.environ.get("MATCHER_DATASET", "recrutas/matcher")
INBOX = "inbox/candidates.json"
OUTBOX = "outbox/latest.json.gz"


def main() -> int:
    from huggingface_hub import HfApi, hf_hub_download
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    inbox = json.loads(Path(hf_hub_download(MATCHER_DATASET, INBOX, repo_type="dataset", force_download=True)).read_text())
    vectors = Path(hf_hub_download(JOBS_DATASET, "vectors/jobs.parquet", repo_type="dataset"))
    state = Path(hf_hub_download(JOBS_DATASET, "state/jobs.parquet", repo_type="dataset"))
    result = run(vectors, state, inbox)
    print(json.dumps(result["stats"]))
    out = Path(tempfile.mkdtemp()) / "latest.json.gz"
    payload = {"generated_at": datetime.now(UTC).isoformat(), "inbox_exported_at": inbox.get("exported_at"),
               "jobs": result["jobs"], "closed": result["closed"], "stats": result["stats"]}
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
