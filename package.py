"""Build submission.zip (guide submission format) and check it.

Usage: python package.py [--templates data/test/csv] [--out submission.zip]

Layout: output/level1.csv, output/level2.csv, source/ (code, tests, requirements, README, .env.example, data/mock_dev),
demo/ (only if demo/demo.mp4|mov|webm exists; warns under 500 KB).
Fails on: >15 MB, a .env file, an API-key pattern or .env key value, a test-bill ID in a shipped source file, a model
string outside the allowed list, a non-source file type, wrong CSV headers, or invalid CSVs.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 15 * 1024 * 1024
SOURCE_GLOBS = ("*.py", "tests/*.py", "scripts/*.py", "data/mock_dev/*", ".streamlit/config.toml")
SOURCE_FILES = ("requirements.txt", "README.md", ".env.example")
EXCLUDED_PARTS = {".env", ".cache", "venv", ".venv", "output_dev", "_prep", "__pycache__", ".claude", ".agents", ".codex"}
EXCLUDED_NAMES = {"CLAUDE.md", "AGENTS.md", "progress.md", "skills-lock.json", ".env", "submission.zip"}
EXCLUDED_PREFIXES = ("data/train/", "data/test/")
# Dev-only checks that need unshipped data (data/train, data/mock); smoke_scaffold also uses rejected model names on purpose.
EXCLUDED_FILES = {"tests/smoke_scaffold.py", "tests/check_extract.py"}
SOURCE_EXTS = {".py", ".md", ".txt", ".toml", ".csv", ".example"}
DEMO_EXTS = (".mp4", ".mov", ".webm")
DEMO_MIN_BYTES = 500 * 1024
L1_HEADER, L2_HEADER = "bill_id,json", "bill_id,question_id,question,answer"
MODEL_RE = re.compile(rb"(?i)\b(?:gemini|gpt|claude|o[1-9])-[a-z0-9][a-z0-9.\-]*")
KEY_PATTERNS = [re.compile(rb"AIza[0-9A-Za-z_\-]{20,}"), re.compile(rb"sk-[A-Za-z0-9_\-]{20,}"),
                re.compile(rb"sk-ant-[A-Za-z0-9_\-]{20,}")]


def _excluded(rel: str) -> bool:
    parts = rel.split("/")
    return (bool(EXCLUDED_PARTS & set(parts[:-1])) or parts[-1] in EXCLUDED_NAMES or rel.startswith(EXCLUDED_PREFIXES)
            or parts[-1].endswith((".pyc", ".pyo")) or rel in EXCLUDED_FILES)


def source_files() -> list[Path]:
    """Repo files shipped under source/ (relative to ROOT), sorted."""
    found = {p for g in SOURCE_GLOBS for p in ROOT.glob(g) if p.is_file()}
    found |= {ROOT / f for f in SOURCE_FILES if (ROOT / f).is_file()}
    return sorted(p for p in found if not _excluded(p.relative_to(ROOT).as_posix()))


def entries() -> list[tuple[Path, str]]:
    """(local path, archive name) pairs."""
    out = [(ROOT / "output" / n, f"output/{n}") for n in ("level1.csv", "level2.csv")]
    missing = [str(p) for p, _ in out if not p.is_file()]
    if missing:
        raise SystemExit(f"Error: missing output file(s): {', '.join(missing)}")
    out += [(p, "source/" + p.relative_to(ROOT).as_posix()) for p in source_files()]
    demo = ROOT / "demo"
    if demo.is_dir():
        out += [(p, f"demo/{p.name}") for p in sorted(demo.iterdir()) if p.is_file() and p.suffix.lower() in DEMO_EXTS]
    return out


def _env_key_values() -> list[bytes]:
    """Actual key values from .env (never printed), to make sure none were copied into a shipped file."""
    env = ROOT / ".env"
    if not env.is_file():
        return []
    vals = []
    for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        if "KEY" in k.upper() and len(v) >= 16:
            vals.append(v.encode())
    return vals


def scan_secrets(zip_path: Path) -> list[str]:
    problems, secrets = [], _env_key_values()
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            if base == ".env" or (base.startswith(".env") and base != ".env.example"):
                problems.append(f"{name}: .env file")
            data = z.read(name)
            if any(p.search(data) for p in KEY_PATTERNS):
                problems.append(f"{name}: API-key pattern")
            if any(s in data for s in secrets):
                problems.append(f"{name}: contains a key value from .env")
            if _excluded(name.split("/", 1)[-1]) and not name.startswith("output/"):
                problems.append(f"{name}: excluded path")
    return problems


def test_bill_ids(templates: Path) -> set[str]:
    """Test bill IDs from data/test/bills and the test templates (computed at run time, never stored in code)."""
    ids = {p.stem for p in (ROOT / "data" / "test" / "bills").glob("*") if p.is_file()}
    t1 = templates / "level1.csv"
    if t1.is_file():
        ids |= {line.split(",", 1)[0].strip() for line in t1.read_text(encoding="utf-8-sig").splitlines()[1:] if line.strip()}
    return {i for i in ids if i}


def audit(zip_path: Path, templates: Path) -> list[str]:
    """Hard-coding, model-string, file-type and CSV-header checks on the built zip."""
    from config import ALLOWED_MODELS
    allowed = {m.lower() for m in ALLOWED_MODELS}
    ids = test_bill_ids(templates)
    problems = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            data = z.read(name)
            if name.startswith("output/"):
                continue
            if name.startswith("source/"):
                ext = Path(name).suffix.lower() or Path(name).name
                if ext not in SOURCE_EXTS:
                    problems.append(f"{name}: not a source file type")
                text = data.decode("utf-8", errors="ignore")
                hits = sorted(i for i in ids if re.search(rf"\b{re.escape(i)}\b", text))
                if hits:
                    problems.append(f"{name}: references test bill(s) {', '.join(hits)}")
                bad = sorted({m.decode().lower() for m in MODEL_RE.findall(data)} - allowed)
                if bad:
                    problems.append(f"{name}: model string(s) not in the allowed list: {', '.join(bad)}")
        for name, header in (("output/level1.csv", L1_HEADER), ("output/level2.csv", L2_HEADER)):
            first = z.read(name).decode("utf-8").splitlines()[0]
            if first != header:
                problems.append(f"{name}: header {first!r} != {header!r}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Build and check submission.zip")
    ap.add_argument("--templates", type=Path, default=ROOT / "data" / "test" / "csv", help="templates to validate against")
    ap.add_argument("--out", type=Path, default=ROOT / "submission.zip")
    args = ap.parse_args()

    files = entries()
    if args.out.exists():
        args.out.unlink()
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path, arc in files:
            z.write(path, arc)

    print(f"{'size':>10}  file")
    for path, arc in files:
        print(f"{path.stat().st_size:>10,}  {arc}")
    size = args.out.stat().st_size
    print(f"\n{len(files)} files, zip size {size:,} bytes ({size / 1024 / 1024:.2f} MB, limit 15 MB)")
    if size > MAX_BYTES:
        print("FAIL: zip is over 15 MB", file=sys.stderr)
        return 1

    problems = scan_secrets(args.out)
    if problems:
        print("FAIL: secrets/excluded files found:\n - " + "\n - ".join(problems), file=sys.stderr)
        return 1
    print("no secrets found")

    problems = audit(args.out, args.templates)
    if problems:
        print("FAIL: audit:\n - " + "\n - ".join(problems), file=sys.stderr)
        return 1
    print("audit OK: no test-bill IDs in source, only allowed model IDs, source file types only, CSV headers exact")
    for path, arc in files:
        if arc.startswith("demo/") and path.stat().st_size < DEMO_MIN_BYTES:
            print(f"WARNING: {arc} is only {path.stat().st_size:,} bytes (< 500 KB); check the recording")
    if not any(arc.startswith("demo/") for _, arc in files):
        print("demo/: none (optional)")

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(args.out) as z:
            z.extractall(tmp, members=[n for n in z.namelist() if n.startswith("output/")])
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        r = subprocess.run([sys.executable, str(ROOT / "tests" / "validate_outputs.py"), str(args.templates),
                            str(Path(tmp) / "output")], capture_output=True, text=True, encoding="utf-8", env=env)
        last = (r.stdout.strip().splitlines() or [""])[-1]
        print(f"validate (zipped CSVs vs {args.templates}): {last}")
        if r.returncode != 0:
            print(r.stdout + r.stderr, file=sys.stderr)
            return 1
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
