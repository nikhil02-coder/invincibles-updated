# Deploying INVINCIBLES to Render

This repo is a monorepo: `backend/` (FastAPI) and `frontend/` (static React
build). Render needs to know that — the single most common cause of
"Exited with status 1" on a build like this is Render trying to run
`pip install -r requirements.txt` from the **repo root**, where no
`requirements.txt` exists (it lives in `backend/`).

## Recommended: deploy via Blueprint (render.yaml)

This repo includes a `render.yaml` at the repo root that defines both
services correctly (root directories, build/start commands, Python
version). Using it avoids all manual-config mistakes.

1. Push this repo to GitHub.
2. In the Render dashboard: **New +** → **Blueprint**.
3. Connect the repo. Render will read `render.yaml` and propose two
   services: `invincibles-api` (Python web service) and
   `invincibles-frontend` (static site).
4. Before deploying, set the `ANTHROPIC_API_KEY` environment variable on
   `invincibles-api` if you want the interactive Lunar AI chat backed by
   Claude (optional — it works without this too, via a rule-based
   fallback).
5. Deploy. First backend build takes a few minutes (installs numpy/scipy/
   opencv/scikit-image).
6. Once both services are live, the frontend's `VITE_API_BASE_URL` in
   `render.yaml` points at `https://invincibles-api.onrender.com` — if
   your API service ends up with a different auto-generated hostname,
   update that value (or better, override it via a static-site env var in
   the dashboard) and redeploy the frontend.

## Alternative: two manually-configured Web Services

If you're not using Blueprint, create the backend service by hand and
**set these exactly**:

| Setting | Value |
|---|---|
| Root Directory | `backend` |
| Runtime | Python 3 |
| Build Command | `pip install --upgrade pip && pip install -r requirements.txt` |
| Start Command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| Environment Variable | `PYTHON_VERSION` = `3.11.11` |
| Health Check Path | `/api/health` |

**Root Directory is the setting people miss most often** — without it,
the build command can't find `requirements.txt` and fails immediately.

For the frontend, create a Static Site:

| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Build Command | `npm ci && npm run build` |
| Publish Directory | `dist` |
| Environment Variable | `VITE_API_BASE_URL` = your backend's URL |
| Redirect/Rewrite Rule | `/*` → `/index.html` (Rewrite) |

The rewrite rule is required because this is a client-side-routed React
app (React Router) — without it, refreshing on any page other than `/`
returns a 404 from Render's static host.

## Why the build was failing

Three concrete, fixed issues:

1. **`requirements.txt` pinned very recent package versions** with no
   guarantee of prebuilt wheels for whatever Python version Render's
   build environment defaults to. If pip can't find a wheel, it tries to
   compile from source, which fails or times out on Render's free build
   tier. Fixed by pinning to older, extremely well-established versions
   (verified in a clean venv, and verified the app actually **boots**
   with them, not just installs) and by pinning `PYTHON_VERSION`
   explicitly via the render.yaml/dashboard mechanism Render currently
   documents (not the older `runtime.txt` convention, which may not take
   effect depending on your Render account's build image).
2. **`uvicorn[standard]`** pulled in `uvloop`/`httptools`/`watchfiles` —
   unnecessary compiled dependencies for a straightforward deployment.
   Switched to plain `uvicorn`.
3. **Dev-only dependencies** (`pytest`, `httpx`) were in the production
   `requirements.txt`. Moved to `requirements-dev.txt` (`pip install -r
   requirements-dev.txt` for local test runs; production only installs
   `requirements.txt`).

## Verifying locally before you redeploy

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/api/health
# should print: {"project":"INVINCIBLES","status":"READY"}
```

If that works locally with a fresh venv on Python 3.11, the same
`requirements.txt` will build on Render as long as Root Directory and
PYTHON_VERSION are set as above.
