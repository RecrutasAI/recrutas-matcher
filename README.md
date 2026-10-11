# recrutas-matcher (private)

Matches Recrutas users to Fleet jobs on GitHub's machines, so the Recrutas server stays light.

1. The Recrutas server writes `inbox/candidates.json` to the private `recrutas/matcher` dataset:
   each user's resume vector under an anonymous id (a keyed hash only the server can map back),
   and the Fleet jobs Recrutas already shows.
2. Every 4 hours `Match` compares those vectors with every open Fleet job (`recrutas/us-jobs`:
   `vectors/jobs.parquet`, made by the Fleet's "Job vectors" workflow) and writes
   `outbox/latest.json.gz`: each user's top 200 jobs with details and vectors, plus shown jobs
   that have closed.
3. The server inserts the new jobs (vectors attached, so it never embeds them) and closes the
   closed ones. Recrutas's feed applies all its own rules to whatever arrives.

Nothing personal goes in logs or artifacts. Tests: `pip install -e ".[dev]" && pytest`.
