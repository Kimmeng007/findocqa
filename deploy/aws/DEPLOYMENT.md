# Deployment (documented, not executed)

**Status: this describes how the project would be deployed to AWS — it
has not actually been deployed.** That's a deliberate scope decision, not
an oversight: AWS costs real, ongoing money (S3, ECS/Fargate, CloudWatch)
unlike everything else in this project, which runs entirely on free
tiers (Gemini free tier, MongoDB Atlas free tier, Streamlit Community
Cloud, GitHub Actions free minutes). For a portfolio project, "here's
exactly how I'd deploy this and why I haven't" is a credible, honest
claim; actually running paid infrastructure indefinitely just to have a
live demo link isn't a cost this project takes on — especially now that
a free alternative already exists and works (see below).

## What exists today

The Streamlit demo (`src/findocqa/ui/`) is real, built, and **already
deployed live** on Streamlit Community Cloud for free — see the README's
live-demo link. There is no separate FastAPI backend
(`src/findocqa/api/` was a placeholder, removed once it became clear
Streamlit covers the same need). The Dockerfile at `deploy/docker/`
containerizes the CLI pipeline (`scripts/ingest.py`,
`scripts/build_index.py`, `scripts/ask.py`, `scripts/agent_ask.py`,
`scripts/run_eval.py`) and is genuinely verified — built, tested
(`pytest` passes inside it), and run end-to-end against the real
MongoDB/Gemini APIs (see `TECHNICAL_REPORT.md` §9) — it's just not
deployed anywhere; it's containerized for portability/reproducibility,
not because anything currently needs it running.

## How it would map to AWS, if it ever did move off Streamlit Cloud

- **S3**: store raw downloaded filings (`data/raw/`) instead of local
  disk, so ingestion is stateless across container restarts/scaling.
- **ECS/Fargate**: run the Streamlit app (using this repo's Dockerfile
  as a base, adapted to serve Streamlit instead of the CLI) as a
  container, auto-scaled behind an Application Load Balancer. Fargate
  over EC2 specifically to avoid managing servers for a low-traffic
  portfolio demo.
- **MongoDB**: already cloud-hosted (Atlas free tier) — no change needed,
  just ensure the ECS task's security group can reach it (Atlas Network
  Access already allows `0.0.0.0/0` per this project's current setup —
  in a real deployment this would be tightened to the ECS task's specific
  outbound IP range instead).
- **Search index**: currently cached in MongoDB Atlas via GridFS
  (`retrieval/index_cache.py`) specifically so any fresh container —
  Streamlit Cloud or otherwise — can download a ready index instead of
  rebuilding from scratch. This already works the way an ECS deployment
  would need it to; no AWS-specific change required here.
- **CloudWatch**: container logs (stdout/stderr) and basic metrics
  (request latency, error rate) — ECS sends these to CloudWatch by
  default when configured with the `awslogs` log driver.
- **Secrets** (`GOOGLE_API_KEY`, `MONGODB_URI`): AWS Secrets Manager,
  injected into the ECS task definition as environment variables at
  runtime — never baked into the Docker image, matching how this project
  already keeps secrets out of `.env` in version control.

## Why not deploy now

1. **Cost**: Fargate + ALB + CloudWatch + S3, even at low traffic, is a
   real recurring bill — not justified for a portfolio piece that gets
   occasional recruiter/interviewer traffic, versus the current
   free-tier-only setup that costs nothing to keep running indefinitely.
2. **A free alternative already works**: Streamlit Community Cloud
   already serves the exact purpose AWS would here (a live, public demo
   link) at zero cost. Moving to AWS would add cost and operational
   complexity (VPC, IAM, load balancers) without adding a capability the
   project currently needs.
3. **The free-tier Gemini quota is the real bottleneck anyway**: even a
   perfectly deployed backend is still bounded by the same 500
   requests/day limit hit twice during this project's own development —
   deploying to AWS doesn't remove that constraint; only a paid Gemini
   tier would, which is a separate cost decision independent of hosting.
