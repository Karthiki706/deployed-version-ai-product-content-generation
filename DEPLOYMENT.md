# Deployment Guide (free tier, ~10 minutes)

```
Recruiter's browser ──▶ Streamlit Community Cloud (frontend)
                              │  X-API-Key header
                              ▼
                        Render web service (FastAPI + LangGraph agents) ──▶ Gemini API
```

You need: a GitHub account, a free [Gemini API key](https://aistudio.google.com/apikey), a [Render](https://render.com) account, and a [Streamlit Community Cloud](https://share.streamlit.io) account.

## 1. Push to GitHub
```bash
git init && git add . && git commit -m "Agentic product content platform"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git && git push -u origin main
```
`.env` and `.streamlit/secrets.toml` are git-ignored, so no secrets are pushed. Check this before pushing.

## 2. Deploy the backend on Render
1. Render dashboard → **New → Blueprint** → pick your repo (it reads `render.yaml`).
2. When prompted, paste your **GEMINI_API_KEY**.
3. Click **Apply** and wait for the Docker build (about 3-5 min).
4. Open `https://<service>.onrender.com/health`. You should see JSON with `"gemini": "connected"`.
5. Render → service → **Environment** → copy the generated **APP_API_KEY** value (needed next).

## 3. Deploy the frontend on Streamlit Community Cloud
1. **New app** → your repo, branch `main`, main file `streamlit_app.py`.
2. **Advanced settings** → Python **3.12**, and paste in Secrets:
   ```toml
   BACKEND_URL = "https://<service>.onrender.com"
   APP_API_KEY = "<value copied from Render>"
   ```
3. Deploy. The sidebar should show the backend as online.

## 4. Verify
- Run a sample product, then open the **Agent Trace** panel. You should see real tool calls.
- Without the key, the API refuses requests:
  `curl -X POST https://<service>.onrender.com/generate -d '{}'` returns `401`.

## Things to know about free tiers
- **Cold starts:** Render free services sleep after about 15 min idle, so the first request can take 30-60 s. Fix: create a free [UptimeRobot](https://uptimerobot.com) monitor that hits `/health` every 5 min. Also open the app yourself before sharing the link with a recruiter.
- **Ephemeral disk:** saved reports in `outputs/` vanish on restart. That is fine for a demo.
- **Cost protection:** `APP_API_KEY` blocks outsiders from calling your API directly, and `RATE_LIMIT_PER_MINUTE` (default 10) caps total Gemini calls. A single run uses several LLM calls, so also set a quota or budget alert in Google AI Studio / Cloud.
- **Model:** `render.yaml` sets `GEMINI_MODEL=gemini-2.5-flash`. Change it in the Render dashboard if you prefer another model. The fallback chain still applies.
- **Rotate keys** if you ever commit one by accident.

## Local run with Docker
```bash
cp .env.example .env      # add GEMINI_API_KEY
docker compose up --build  # API :8000, UI :8501
```

## Resume / README touches
Add under the title of your README, and in your resume:
```
Live demo: https://<your-app>.streamlit.app   |   API docs: https://<service>.onrender.com/docs
```
