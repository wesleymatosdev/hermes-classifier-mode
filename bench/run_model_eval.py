#!/usr/bin/env python3
"""Per-model eval of the LLM classifier layer ONLY (no static fast paths, no cache).

Measures, for each model named on argv:
  - malicious recall  (CommandInjection set: verdict should be block)
  - benign FP rate    (nl2bash set: verdict should be allow)
  - real-hermes FP    (anonymized real commands: verdict should be allow)
  - latency (median/mean) per model verdict

Writes bench/results_model/<slug>.json and prints a comparison table.
Usage: python3 bench/run_model_eval.py MODEL[:tag] [MODEL2 ...] [--sample-real N]
"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
import importlib.util

_spec = importlib.util.spec_from_file_location("hcm", ROOT / "__init__.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

# reuse dataset loaders from benchmark.py
_spec2 = importlib.util.spec_from_file_location("benchmark", ROOT / "bench" / "benchmark.py")
bm = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(bm)

OUT_DIR = Path(__file__).resolve().parent / "results_model"


def eval_model(model: str, mal, benign, real, workers=4, timeout_s=20, kinds="mal,ben,real"):
    # fresh cache per model so nothing leaks across models
    cm._VERDICT_CACHE.clear()
    cfg = dict(cm._DEFAULTS)
    cfg["model"] = model
    cfg["verdict_cache_s"] = 0  # measure the model, not the cache
    cfg["timeout_s"] = timeout_s

    kinds = kinds.split(",")
    jobs = ([( "mal", "block", c) for c in mal if "mal" in kinds]
            + [("ben", "allow", c) for c in benign if "ben" in kinds]
            + [("real", "allow", c) for _, c in real if "real" in kinds])

    def run(job):
        kind, want, cmd = job
        t0 = time.perf_counter()
        v = cm._ollama_classify(cfg, cmd)
        dt = time.perf_counter() - t0
        verdict = v["verdict"] if v else "fail"
        return (kind, want, verdict, dt, cmd)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(run, jobs))
    return rows


def summarize(rows):
    out = {}
    for kind in ("mal", "ben", "real"):
        sub = [r for r in rows if r[0] == kind]
        if not sub:
            continue
        n = len(sub)
        want = sub[0][1]
        got_want = sum(1 for r in sub if r[2] == want)
        fails = [r for r in sub if r[2] != want]
        lat = [r[3] for r in sub]
        out[kind] = {
            "n": n, "want": want,
            f"correct": got_want,
            "rate": round(got_want / n, 4),
            "fail_verdicts": {},
            "lat_median_s": round(statistics.median(lat), 3),
            "lat_mean_s": round(statistics.fmean(lat), 3),
            "lat_p90_s": round(sorted(lat)[int(0.9 * len(lat))], 3),
        }
        for r in fails:
            out[kind]["fail_verdicts"][r[2]] = out[kind]["fail_verdicts"].get(r[2], 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--sample-real", type=int, default=400)
    ap.add_argument("--sample-benign", type=int, default=300)
    ap.add_argument("--sample-mal", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=20, help="per-verdict timeout (s)")
    ap.add_argument("--out", default=None,
                    help="override output filename under bench/results_model/")
    ap.add_argument("--kinds", default="mal,ben,real",
                    help="comma list of dataset kinds to include (chunking aid)")
    args = ap.parse_args()

    mal = bm.load_malicious()
    if args.sample_mal:
        mal = mal[: args.sample_mal]
    benign = bm.load_benign(args.sample_benign)
    real = bm.load_hermes_real(None)
    if args.sample_real and len(real) > args.sample_real:
        import random
        random.seed(42)
        real = random.sample(real, args.sample_real)

    print(f"dataset: {len(mal)} mal | {len(benign)} benign | {len(real)} real\n", flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    table = {}
    for model in args.models:
        slug = model.replace("/", "_").replace(":", "_")
        print(f"=== {model} ===", flush=True)
        rows = eval_model(model, mal, benign, real, workers=args.workers,
                          timeout_s=args.timeout, kinds=args.kinds)
        s = summarize(rows)
        table[model] = s
        for kind, label in (("mal", "malicious recall"), ("ben", "benign allow"),
                            ("real", "real allow")):
            if kind in s:
                k = s[kind]
                print(f"  {label:<18} {k['rate']:>7.1%}  ({k['correct']}/{k['n']})"
                      f"  fails={k['fail_verdicts']}  med={k['lat_median_s']}s", flush=True)
        # persist per-command rows for FP eyeballing
        out_name = args.out or f"{slug}.json"
        with open(OUT_DIR / out_name, "w") as f:
            json.dump({"model": model, "summary": s,
                       "rows": [{"kind": r[0], "want": r[1], "verdict": r[2],
                                 "dt": round(r[3], 3), "cmd": r[4][:300]} for r in rows]},
                      f, indent=1)
    print("\n==== COMPARISON ====")
    print(f"{'model':<50} {'mal-rec':>8} {'ben-allow':>10} {'real-allow':>11} {'med-lat':>8}")
    for model, s in table.items():
        print(f"{model:<50} "
              f"{s.get('mal', {}).get('rate', 0):>8.1%} "
              f"{s.get('ben', {}).get('rate', 0):>10.1%} "
              f"{s.get('real', {}).get('rate', 0):>11.1%} "
              f"{s.get('mal', {}).get('lat_median_s', 0):>7.2f}s")
    with open(OUT_DIR / "comparison.json", "w") as f:
        json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()
