# Eval

Measures how accurately BillAsaan reads a bill (Level 1 fields, charge and tax lines) and whether chat answers
state the numbers that `calc.py` computes. Results: [RESULTS.md](RESULTS.md).

## Data (private, gitignored)

| Path | What | Committed? |
|---|---|---|
| `eval/bills/<bill_id>.png\|jpg` | Real bill images | No |
| `eval/labels/<bill_id>.json` | Hand-verified labels | No |
| `eval/RESULTS.md` | Aggregate scores only, no bill values | Yes |

`KESC_0008` is seeded automatically as a verified label from `data/mock/kesc_0008_expected.json`; its image is read
from `data/train/bills/` if it isn't in `eval/bills/`. It was used while developing the prompt, so it is not held out.

## Workflow

```bash
# 1. Put bill images in eval/bills/, then write draft labels from the current extraction
python eval/run_eval.py --draft

# 2. Verify EVERY field of every draft against its image (see below), then set "verified": true

# 3. Score verified labels and write eval/RESULTS.md
python eval/run_eval.py              # add --no-chat to score extraction only
```

Drafts are copies of the model's own output, so the script prints a loud warning and the scorer **skips any label
without `"verified": true`**. Existing labels are never overwritten.

### Verifying a label

- `level1`: correct every field to what the bill prints. Use `null` only when the bill truly doesn't show it.
  Dates `YYYY-MM-DD`, bill month `YYYY-MM`, amounts as plain numbers.
- `charges` / `taxes`: one entry per printed line, `type` from the schema. A single combined tax line means
  `taxes: []` plus `total_taxes` (task spec §5.2).
- `chat[].expected`: the value `calc.py` should produce from the corrected bill (e.g. `late_surcharge` =
  payable after − payable within; `prev12_avg_units` = average of the printed previous months). `null` = not scored.

## Metrics

| Metric | Definition |
|---|---|
| Field accuracy | Per bill and field: correct if both null, text/dates equal (trimmed), or numbers within ±1 |
| Invented | Label is `null`, extraction gave a value |
| Missed | Label has a value, extraction gave `null` |
| Wrong | Both have values that don't match |
| Line items | Charges and taxes paired by `type` in printed order; correct if amount within ±1. Precision = correct / extracted, recall = correct / labelled |
| Chat number accuracy | 5 fixed questions per bill; correct if the answer contains the expected value (exact, or rounded to 0–2 decimals) |

## Cost

Extractions reuse the `run.py` disk cache (`.cache/extract/`, keyed on image + model + extraction settings) and chat
answers are cached in `.cache/eval_chat/` (keyed on model + chat prompt + context + question). Reruns make **no API
calls**; a new bill costs 1 extraction + up to 5 chat calls, paced at one call per 5 s to stay inside the
free-tier per-minute limit. Failed calls are never cached.
