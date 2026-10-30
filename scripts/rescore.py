"""Re-apply the (reference-free) guardrails and gender scorer to stored outputs.

Model outputs are saved verbatim by src.evaluate, so detector improvements can be applied
without re-running any model:  python scripts/rescore.py && python -m src.report
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import guardrails as G                     # noqa: E402
from src.config import RESULTS_DIR                  # noqa: E402
from src.probes import all_probes, score_gender     # noqa: E402

for mpath in sorted((RESULTS_DIR / "metrics").glob("*.json")):
    key = mpath.stem
    tpath = RESULTS_DIR / "translations" / f"{key}.jsonl"
    rows = [json.loads(l) for l in tpath.open()]
    reps = []
    for r in rows:
        rep = G.check(r["src"], r["hyp"], r["sim"])
        r.update(rep.as_dict())
        reps.append(rep)
    tpath.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    m = json.loads(mpath.read_text())
    m["flag_rates"] = {k: float(np.mean([x.flags[k] for x in reps]) * 100) for k in reps[0].flags}
    m["hallucination_rate"] = float(np.mean([x.hallucination for x in reps]) * 100)
    mpath.write_text(json.dumps(m, indent=1, ensure_ascii=False))
    print(f"{key:18s} flagged {m['hallucination_rate']:.1f}%")

items = all_probes()
for ppath in sorted((RESULTS_DIR / "probes").glob("*.json")):
    p = json.loads(ppath.read_text())
    for it in p["items"]:
        it.update(G.check(it["en"], it["hi"], it["sim"]).as_dict())
    outs = {it["id"]: it["hi"] for it in p["items"]}
    p["gender"] = score_gender(items, [outs[i["id"]] for i in items])
    ppath.write_text(json.dumps(p, indent=1, ensure_ascii=False))
    print(f"{ppath.stem:18s} gender acc {p['gender']['accuracy']:.1f}  undetermined {p['gender']['undetermined_pct']:.1f}%")
