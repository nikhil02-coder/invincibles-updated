# INVINCIBLES

### Multi-Modal, Sun-Angle and Scale-Invariant Lunar Image Registration
**Full-stack prototype — Smart India Hackathon**

Real, executable, scientifically traceable registration of multi-sensor
Chandrayaan-2-style lunar observations (TMC-Azimuth, TMC-Slope, IIRS, SAR)
into a common OHRC reference frame, with a scientific-workstation frontend
on top. Every number and every image shown anywhere in the UI is computed
or rendered by the backend from the actual images in `backend/data/raw/`
— nothing is fabricated, including the overall composite registration
image and the original-vs-registered comparison (see `backend/README.md`
§4.5).

```
INVINCIBLES/
├── render.yaml       Render Blueprint (see DEPLOYMENT.md)
├── DEPLOYMENT.md      Render deployment guide
├── backend/           FastAPI + the full registration engine (see backend/README.md)
└── frontend/          React + TypeScript mission-control UI (see frontend/README.md)
```

## Run the whole stack locally

**1. Backend** (do this first — the frontend has nothing to show without it):

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # production deps only
# pip install -r requirements-dev.txt    # use this instead if you also want to run pytest
uvicorn api.main:app --reload
```

This starts the API at `http://localhost:8000`. Optionally set
`ANTHROPIC_API_KEY` in your environment first to enable the interactive
Lunar AI chat backed by Claude (it works fully without it too, via a
broad rule-based fallback — see `backend/README.md` §6).

**2. Frontend**, in a second terminal:

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open `http://localhost:5173`. Click **RUN FULL ANALYSIS** on Mission
Control to kick off a real registration run against the real dataset —
first run takes roughly 30–90 seconds for all four source sensors; repeat
runs with identical settings return instantly from the backend's cache.

## Deploying

See **`DEPLOYMENT.md`** for a Render deployment guide (Blueprint via
`render.yaml`, or manual two-service setup) — it documents exactly what
causes the most common "Exited with status 1" build failure on this kind
of monorepo and how this repo avoids it.

## What's already been verified end-to-end

- `cd backend && pip install -r requirements-dev.txt && pytest tests/ -v`
  — 36/36 passing, including a regression test that all four source
  sensors (TMC-Azimuth, TMC-Slope, IIRS, SAR) reach `SUCCESS` against
  OHRC on the real supplied dataset, and tests for the composite-image
  generation and SSIM-based comparison metrics.
- `cd frontend && npm run build` — clean TypeScript build, no errors.
- `backend/requirements.txt` installs cleanly into a fresh virtualenv and
  the app boots and serves `/api/health` with those exact production-only
  dependencies (no dev deps, no unused extras) — this is what a Render
  build actually runs.
- A full run was triggered through the live UI against the live backend
  and every page (Mission Control, Dataset, Registration, Feature
  Analysis, Correspondence, Results, Multi-Sensor incl. the composite
  overview and original-vs-registered comparison grid, Lunar AI incl.
  region click + chat, Reports, Run History) was screenshotted and
  inspected against the real API responses.

## Design note

The frontend was built strictly backend-first: the backend's actual
endpoints, response shapes, and behavior were read directly from source
before any UI code was written (see `frontend/README.md` for the mapping
and the handful of small, additive, non-algorithmic backend endpoints that
were added purely to expose data the frontend needed — e.g. run history,
resolved diagnostic-image URLs, the comparison endpoint — documented in
`backend/README.md` §6-7). No registration algorithm code was changed to
build the frontend or to fix the deployment.
