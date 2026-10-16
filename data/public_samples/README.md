# Public samples

No verified anonymized sample images are currently provided. Never point this interface at original train/test bills.

After a human verifies that an image has no names, addresses, account/reference/meter numbers, barcodes, QR codes, or other identifying details, place only that anonymized copy here. A second review should check the rendered image and metadata. Register the reviewed bytes in `manifest.json`, a JSON list of objects:

```json
[
  {
    "file": "anonymized-example.png",
    "label": "Anonymized example",
    "verified_anonymized": true,
    "reviewed_by": "reviewer identifier",
    "sha256": "SHA-256 of the reviewed file bytes"
  }
]
```

Only direct JPG/PNG files inside this directory with explicit review attestation and matching hashes are eligible. Missing, malformed, unreviewed, changed, or out-of-directory entries are hidden. The manifest is an approval record; it does not automatically detect identifying details. Keep this list empty until review is complete.
