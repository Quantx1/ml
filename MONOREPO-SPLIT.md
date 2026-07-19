# Monorepo Split — What Was Done & How to Run Everything (2026-07-19)

This monorepo was split into 4 repos under the **[Quantx1](https://github.com/Quantx1)** GitHub org:

| Repo | URL | Contents |
|------|-----|----------|
| landing | https://github.com/Quantx1/landing | Marketing/legal site: `/`, `/pricing`, `/privacy`, `/terms`, `/proof` |
| frontend | https://github.com/Quantx1/frontend | Product app (signals, stocks, strategies, trades, copilot, auth, admin…) |
| backend | https://github.com/Quantx1/backend | FastAPI backend + `ml` as a **git submodule** + `artifacts/`, `data/`, `scripts/`, `infrastructure/`, `docs/`, `supabase/`, tests |
| ml | https://github.com/Quantx1/ml | The `ml` Python package (features, trainers, regime, backtest, eval) + its tests |

This monorepo (`Ri2506/quantx`) was left **untouched** as the historical archive. All new repos started as fresh single commits (no history carried over).

---

## How the cross-repo dependencies were solved

- **backend → ml** (4 modules import `ml.*`): ml is a **git submodule at `./ml`** inside backend, so imports resolve exactly like the monorepo. Always clone backend with `--recurse-submodules`.
- **ml → backend** (data_loader needed 2 provider files): those files are **vendored** into `ml/_vendor/backend/data/providers/` (`base.py`, `free_provider.py`). ml is fully standalone. ⚠️ If you edit those 2 files in backend, re-copy them into ml's `_vendor/`.
- **frontend/landing → backend**: runtime-only via `NEXT_PUBLIC_API_URL` env var (HTTP), no code dependency.
- **landing** was carved out of frontend by tracing the full import graph of the 5 marketing routes (133 files). Those routes and their landing-only components were **removed from frontend**; frontend's `/` now redirects to `/copilot`.

## Fixes made during the split (already pushed)

1. backend: added `scripts/` (production code imports `scripts.*`) and `infrastructure/` (migration SQL used by tests).
2. All repos: added `.gitignore` (monorepo only had one at root).
3. ml: 6 test files that tested backend code (providers, serving engines) moved back to backend's `tests/ml/`.
4. ml: 4 trainer files (`momentum/swing/positional_lambdarank.py`, `meta_conviction.py`) got a `_ROOT` fallback so they work standalone; `data/nse_tiers/` is shipped in the ml repo.
5. landing: added `package-lock.json`, `.env.example`, `scripts/force-wasm.js`; package renamed `quantx-landing`.
6. All repos: README + GitHub Actions CI (`.github/workflows/ci.yml`) adapted to the new layouts; frontend/landing also got the pre-commit guard scripts (hex-literal paths adapted `frontend/app/` → `app/`).

---

## How to run each repo

### backend (FastAPI, port 8000)

```bash
git clone --recurse-submodules https://github.com/Quantx1/backend.git
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/mac: source .venv/bin/activate)
pip install -r requirements.txt
# copy the .env from the old monorepo root (X:\quantx\.env) into this folder — it is gitignored
uvicorn backend.api.app:app --reload --port 8000
# health check: http://localhost:8000/health
```

If you forgot `--recurse-submodules`: `git submodule update --init`.
To update ml inside backend later: `cd ml && git pull origin main && cd .. && git commit -am "bump ml"`.

### frontend (Next.js product app, port 3000)

```bash
git clone https://github.com/Quantx1/frontend.git
cd frontend
npm install
copy .env.example .env.local    # then fill in values (see below)
npm run dev                     # http://localhost:3000  (/ redirects to /copilot)
```

`.env.local` values (Supabase URL/anon key are in the old monorepo's `X:\quantx\.env`):

```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=<from old .env>
NEXT_PUBLIC_SUPABASE_ANON_KEY=<from old .env>
```

### landing (Next.js marketing site)

```bash
git clone https://github.com/Quantx1/landing.git
cd landing
npm install
copy .env.example .env.local    # same values as frontend above
npm run dev                     # runs on :3000 — use "npm run dev -- -p 3001" if frontend is also running
```

### ml (standalone training/research)

⚠️ Repo root **IS** the `ml` package — clone into a folder named `ml` (the default) and run Python from the **parent** directory:

```bash
git clone https://github.com/Quantx1/ml.git    # creates ./ml
pip install -r ml/requirements-train.txt
pytest ml/tests -q                              # run from the parent dir, not inside ml/
```

Optional: put local OHLCV CSVs in `ml/data/cache/` (gitignored) for offline training; otherwise trainers fall back to `data/nse_tiers/` tier lists + yfinance.

---

## What was VERIFIED ✅

- landing: production build green (all 12 pages)
- frontend: production build green (full route table), no dangling imports
- backend: entrypoint + all ml-dependent modules import via submodule; **907/918 tests pass** — the 11 failures fail identically in this monorepo (pre-existing, not split-caused: `test_route_seams`, `test_audit_regressions`, `test_pr_depth`)
- ml: **129/129 tests pass** standalone

## What was NOT verified ❌ (remaining to-do)

1. **Runtime click-through**: dev servers were never started; no real browser request went frontend → backend. Do this first: start backend, start frontend against it, log in, load signals/stocks pages, watch the network tab.
2. **Real-credential flows**: Supabase auth, broker connect, Razorpay (builds used placeholder env values).
3. **CI runs on GitHub**: workflows were pushed but not watched. Set repo secrets first — frontend & landing need `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` (frontend also `RAZORPAY_KEY_ID`) — then check the Actions tabs.
4. **Production deploys** from the new repos (Vercel for landing/frontend at repo root — old `vercel.json` is obsolete; Railway/Nixpacks for backend — enable submodule checkout in the build).
5. The 4 ml training tests needing local `data/cache` CSVs (machine-local, gitignored data).

## Not moved anywhere (deliberate)

- `agent/`, `.agents`, `.mcp.json`, `skills-lock.json` — local Claude Code dev tooling
- `vercel.json` — pointed Vercel at the `frontend/` subfolder; obsolete with apps at repo root
- `deploy.yml`, `release-hardening-gates.yml` — monorepo-wide CI, superseded by per-repo `ci.yml`
- Root `package.json` (only dev playwright), `README.md` (each repo has its own now)
