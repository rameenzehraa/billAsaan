# BillAsaan — Utility Bill Decoder
*Bijli ka bill, ab aasaan.*

Women Tech Quest 2026, Build with AI track. BillAsaan reads Pakistani electricity bill images (IESCO, KE, LESCO), extracts structured billing details (Level 1), and answers customer questions about each bill in plain English (Level 2). Both output CSVs are generated entirely by code.

## Public checkout and private data

The public repository contains code, empty-answer development CSV templates, an empty public-sample manifest, and synthetic offline UI tests. Original train/test bill images, bill-derived expected JSON, generated outputs, API caches/logs, user-upload folders, recordings, submission ZIPs and the organizer PDF stay local and are ignored. They have not been certified anonymous or cleared for redistribution.

A fresh checkout can run the public UI and offline tests without competition data. For the competition CLI, supply your own bill images and matching CSV templates using the paths in the command below. `tests/check_extract.py` and `tests/smoke_scaffold.py` are legacy development helpers that require excluded local competition data; they are not the public offline test suite. `package.py` is the legacy competition packaging tool and requires local competition inputs/outputs; do not publish its generated ZIP as a public source release.

Only `.env.example` is intended for publication; create `.env` locally. Public sample images remain excluded until anonymization review and an explicit Git ignore-rule update. Never use `git add -f` for private data.

## Model declaration
- **Model used in the application:** `gemini-3.5-flash-lite` (Google AI Studio), temperature 0.
- **Fallback (configurable, same allowed list):** `gemini-3.1-flash-lite`.
- No other model, OCR service, or invoice/receipt extraction service is used.

## Setup
Requires Python 3.11+. Developed and tested with **Python 3.13.9**.

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # then put your key in .env
```

`.env` variables:

| Variable | Example | Purpose |
|---|---|---|
| `MODEL_PROVIDER` | `google` | Provider. Must match the model (`google` for Gemini models), otherwise startup fails |
| `MODEL_NAME` | `gemini-3.5-flash-lite` | Must be one of the allowed models, otherwise startup fails. Defaults to `gemini-3.5-flash-lite` |
| `API_KEY` | `...` | Google AI Studio key. Never committed or logged |

The older names `MODEL_ID` (for `MODEL_NAME`) and `GEMINI_API_KEY` (for `API_KEY`) are also accepted as fallbacks.

## Run

**Batch, which generates both CSVs:**
```bash
python run.py --bills <bills_dir> --templates <csv_dir> --out output/
```
- `<bills_dir>` holds the bill images (`.png`, `.jpg`, `.jpeg`); bill_id = file name without extension.
- `<csv_dir>` contains the provided `level1.csv` and `level2.csv` templates.
- Writes `output/level1.csv` and `output/level2.csv` in template order, then prints a per-bill summary (cache hit/miss, tokens, latency, checks) and an `UNVERIFIED ANSWERS` block.
- `--refresh` ignores cached results and calls the model again (successful results are still cached).
- Bad input (missing folder, no images, missing template, missing columns, empty template) prints `Error: <message>` and exits 1.

**Public bill explainer:**
```bash
streamlit run app.py
```
- Choose **Upload my bill** (JPG/PNG, up to 10 MB and 20 million pixels) or **Try a sample**. Previewing and switching bills make no API calls.
- Select **Process** to invoke the existing extraction service. Successful results disable duplicate processing. Failures allow retry. No new API provider is used.
- Review **Bill summary**, **Charges and taxes**, **Usage history**, and **Checks**. Under **Review extracted details**, edit the structured fields and select **Apply corrections**. Corrections are Pydantic-validated; local checks/calculations rerun, while the original extraction is preserved separately. Corrections never change competition CSVs.
- Results and conversation state are session-only and clear when the selected bill changes. Uploads bypass the shared disk cache and extraction debug dumps. Images are re-encoded without metadata before preview and extraction.
- Chat suggestions and submission are disabled with an integration-pending message. API chat integration remains future work; no replies are fabricated.
- Samples require explicit anonymization review and a matching content hash in `data/public_samples/manifest.json` (see that folder's README). No original train/test bills are exposed. Without verified samples, the app shows an empty state.
- The competition CLI and CSV workflow above remain available through `run.py`; batch controls are no longer exposed in the public interface. Existing competition configuration/model guards remain unchanged.

**Offline UI verification (no server or API calls):**
```bash
python -m unittest discover -s tests -p test_public_viewer.py
```
Uses synthetic images and mocked extraction. Covers validation, sample gating, processing, duplicate prevention, corrections, resets, failures, and disabled chat.

**Validate outputs:**
```bash
python tests/validate_outputs.py <csv_dir> output/
```
Checks row counts and order against the templates, one-line JSON that validates against the §5.1 schema with every key, identical bill_id/question_id/question, no empty cells and UTF-8, then reports per-bill check statuses and re-ask/unverified/failed counts.

Other helpers: `python scripts/test_keys.py` (checks each Gemini key in `.env` against the allowed Gemini models; never prints keys). `data/mock_dev/` holds a small development template set for the train bills.

## Architecture

```
bill image ─► images.py (EXIF fix, downscale only if long side > 2000px)
          ─► extract.py  (1 vision call → JSON → Pydantic validation)
          ─► checks.py   (local consistency checks; 1 re-ask on Mismatch or validation failure)
          ─► cache.py    (exact-match cache, successes only)
          ─► calc.py     (code computes all numbers + method strings)
          ─► answer.py   (1 text call per bill: model picks expressions,
                          code evaluates them and fills the answer text)
          ─► run.py      (template-ordered level1.csv / level2.csv)
app.py + viewer.py = session-only public UI over existing extraction, checks and calculations
```

| File | Role |
|---|---|
| `config.py` | Settings from `.env`, allowed-model guard, timeouts/retries, versions (extraction prompt `v3`, answer prompt `v7`, calc `c2`, schema `v1`) |
| `schema.py` | Pydantic models: `Level1Bill` (exact §5.1 fields) and `InternalFacts` (history, rates, splits, late tiers, flags; never exported) |
| `images.py` | Lists bills and loads/resizes images (bill_id = file stem; KESC files map to provider `KE` in `config.py`) |
| `extract.py` | Vision extraction, value-preserving normalisation, retries, truncation handling, one re-ask |
| `checks.py` | Date formats, readings vs units (±1 rounding tolerance for decimal readings), charges + taxes vs current bill, current + arrears vs payable, late ≥ within. Returns Matched / Mismatch / Insufficient Data. Never changes values |
| `calc.py` | Named facts (history window with available-month count, averages and comparisons, seasons, 3-month trends, payment history, tax share, surcharge tiers, cost-per-unit estimates, net-metering evidence) plus a whitelisted `safe_eval` |
| `answer.py` | Answer call, number-trace validation, re-ask, fallbacks |
| `cache.py` | sha256-keyed atomic cache (`.cache/`) |
| `run.py` | CLI, input validation and `run_pipeline()` |
| `app.py`, `viewer.py` | Public bill explainer, private session state and validated uploads |
| `tests/validate_outputs.py` | Output CSV validator |

### Reliability
- Timeouts, and at most 2 retries with backoff on 429/5xx/timeout.
- A failure on one bill never stops the run:
  - If extraction fails, the Level 1 row is written fully keyed with nulls and empty lists, and that bill's answers say "Processing failed: this bill could not be read."
  - If answering fails, the extraction is kept and the failed question says "Processing failed: an answer could not be generated for this question." (distinct from "The bill does not show ...").
- Only successful, validated results are cached. Same inputs + settings + prompt/calc versions → byte-identical CSVs, with no API calls.
- Bad input (missing folders, templates or columns) prints a one-line error and exits 1.
- Every model call logs its tokens and latency.

## AI use
**In the application:**
1. **Extraction (vision).** One call per bill reads the image (tilted scans, Urdu text, varied layouts) into the Pydantic schema. The prompt enforces the guide's rules:
   - values copied as printed, with no rounding or correction
   - credits negative
   - null when not shown
   - combined taxes go to `total_taxes` only
   - personal identifiers excluded

   Code then checks consistency. If a check finds a Mismatch (or the output fails validation), it re-asks once about the flagged fields only. The re-asked result replaces the original only if it validates, has fewer mismatches, and doesn't drop any field the original had filled.
2. **Answers (text).** Code computes every number. The model picks which computed facts and expressions answer each question and writes a sentence template. Code then evaluates the expressions and fills in the numbers and dates. An answer containing a number that doesn't trace back to a computed value triggers one re-ask; if it still can't be traced, it is kept and flagged as unverified. Answers:
   - label estimates and explain the method
   - say when a value isn't on the bill
   - state the interpretation used for ambiguous questions, the history window and how many months were used

**Why this design:** the model handles what code can't, which is reading messy images and understanding the questions. Code handles what the model is bad at: arithmetic and repeatability.

**Limitations:**
- A heavily folded or handwritten-over photo can still be misread.
- A charge printed outside the main box can occasionally be missed. The payable check flags this.
- Net metering can't be confirmed from the extracted fields, so answers about it are hedged.

**AI coding tools used during development:** Claude Code (coding agent); Claude and ChatGPT (chat, for planning and review).

## Rules followed
- Only allowed models, and no banned extraction services.
- No test-bill values in code or data files.
- CSVs are written by code with the `csv` module, in UTF-8, keeping template order and columns. They are never edited by hand.
- `.env`, keys and `.cache/` are excluded from the submission.

## Submission layout
`python package.py` builds `submission.zip`: `output/` (both CSVs), `source/` (code, tests, `requirements.txt`, this README, `.env.example`, `data/mock_dev/`) and, optionally, `demo/` if a `demo.mp4|mov|webm` exists.
