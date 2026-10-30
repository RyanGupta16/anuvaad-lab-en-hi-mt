"""Export the dashboard as a static site (GitHub Pages) into docs/.

The static copy shows every result and replays stored translations; the live arena
needs the FastAPI server (app/server.py).
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
OUT.mkdir(exist_ok=True)
for f in ("index.html", "style.css", "app.js"):
    shutil.copy(ROOT / "app" / "static" / f, OUT / f)
shutil.copy(ROOT / "results" / "summary.json", OUT / "summary.json")
(OUT / "translations").mkdir(exist_ok=True)
for f in (ROOT / "results" / "translations").glob("*.jsonl"):
    shutil.copy(f, OUT / "translations" / f.name)
(OUT / ".nojekyll").touch()
print(f"static dashboard written to {OUT}")
