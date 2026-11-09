# Deploying BillAsaan to Streamlit Community Cloud

1. **Push** this repo to GitHub (public or private). Check first that `git status` shows no `.env`,
   `.streamlit/secrets.toml`, bills, caches or competition files (they are gitignored).
2. Open **https://share.streamlit.io** and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub** (or "New app").
4. Choose **repository**, **branch** (`main`) and **main file path** `app.py`.
5. Open **Advanced settings**:
   - **Python version: 3.12**
   - **Secrets:** paste the block below with your real key.
6. Click **Deploy**. The first build installs `requirements.txt` (a few minutes). No `packages.txt` is
   needed: `pypdfium2` and `pillow-heif` ship Linux wheels with their native libraries.
7. Open the app and check: the consent notice shows, a sample/upload processes, and **System stats**
   (sidebar) counts the call. If the page shows "Configuration error: API_KEY is not set", fix the secrets
   (app menu → **Settings** → **Secrets**) and the app restarts by itself.

## Secrets template

```toml
API_KEY = "your-gemini-api-key"
MODEL_NAME = "gemini-3.5-flash-lite"
FREE_TIER = true
MAX_BILLS_PER_SESSION = 3
MAX_QUESTIONS_PER_BILL = 15
DAILY_EXTRACTION_CAP = 150
DAILY_CHAT_CAP = 400
```

Same keys work locally in `.streamlit/secrets.toml` (gitignored) or `.env`; secrets take precedence.

## Notes

- Daily caps, the sample cache and System stats live in memory: they reset when the app restarts or
  Community Cloud puts it to sleep.
- Keep `FREE_TIER = true` while using a free-tier key: Google may use free-tier content to improve its products,
  and the app asks users to consent and to cover personal details first.
- Rotate the key in Google AI Studio if it is ever exposed, then update the Cloud secret.
