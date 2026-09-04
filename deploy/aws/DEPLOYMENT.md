# Deployment (documented, not executed)

**Status: this describes how the project would be deployed to AWS — it
has not actually been deployed.** That's a deliberate scope decision, not
an oversight: AWS costs real, ongoing money (S3, ECS/Fargate, CloudWatch)
unlike everything else in this project, which runs entirely on free
tiers (Gemini free tier, MongoDB Atlas free tier, GitHub Actions free
minutes). For a portfolio project, "here's exactly how I'd deploy this
and why" is a credible, honest claim; actually running paid infrastructure
indefinitely just to have a live demo link isn't a cost this project
takes on. The Dockerfile at the repo root is real and reviewed, just not
locally build-tested (Docker isn't installed in this project's dev
environment) or deployed.

## What exists today vs. what deployment would front

Today this project is a set of CLI scripts (`scripts/ingest.py`,
`scripts/build_index.py`, `scripts/ask.py`, `scripts/agent_ask.py`,
`scripts/run_eval.py`) plus the underlying `src/findocqa` package. There
is no FastAPI backend or Streamlit UI yet (`src/findocqa/api/` and
`src/findocqa/ui/` are placeholders) — those would be the actual thing
deployed; deploying the current CLI scripts as-is wouldn't make sense as
a "live demo."

## How it would map to AWS

- **S3**: store raw downloaded filings (`data/raw/`) instead of local
  disk, so ingestion is stateless across container restarts/scaling.
- **ECS/Fargate**: run the FastAPI backend (once built) as a container
  (using this repo's `Dockerfile` as the base), auto-scaled behind an
  Application Load Balancer. Fargate over EC2 specifically to avoid
  managing servers for a low-traffic portfolio demo.
- **MongoDB**: already cloud-hosted (Atlas free tier) — no change needed,
  just ensure the ECS task's security group can reach it (Atlas Network
  Access already allows `0.0.0.0/0` per this project's current setup —
  in a real deployment this would be tightened to the ECS task's specific
  outbound IP range instead).
- **FAISS index files**: currently written to local disk
  (`data/processed/{variant}/faiss/`) — would move to an EFS volume
  mounted into the ECS task (or rebuilt from MongoDB on container start
  for a stateless task, trading startup time for simpler infra), since
  they need to persist and be shared across task replicas.
- **CloudWatch**: container logs (stdout/stderr from the FastAPI app) and
  basic metrics (request latency, error rate) — ECS sends these to
  CloudWatch by default when configured with the `awslogs` log driver.
- **Secrets** (`GOOGLE_API_KEY`, `MONGODB_URI`): AWS Secrets Manager,
  injected into the ECS task definition as environment variables at
  runtime — never baked into the Docker image, matching how this project
  already keeps secrets out of `.env` in version control.

## Why not deploy now

1. **Cost**: Fargate + ALB + CloudWatch + S3, even at low traffic, is a
   real recurring bill — not justified for a portfolio piece that gets
   occasional recruiter/interviewer traffic, versus a free-tier-only
   setup that costs nothing to keep running indefinitely.
2. **Nothing to deploy yet**: the FastAPI/Streamlit layer this would
   front doesn't exist as part of this project yet (Week 5's remaining
   scope, not done in this pass).
3. **The free-tier Gemini quota is the real bottleneck anyway**: even a
   perfectly deployed backend is still bounded by the same 500
   requests/day limit hit twice during this project's own development —
   deploying doesn't remove that constraint, only a paid Gemini tier
   would, which is a real cost decision independent of AWS.
