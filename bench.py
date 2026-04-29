#!/usr/bin/env python3
"""bench.py — autoresearch verdict tool.

Run a benchmark command N times, optionally check correctness, compare to a
saved baseline, emit a JSON verdict (KEEP or REVERT).

The agent loop calls this after every code change. The verdict is
authoritative — the agent does not second-guess it.

Configure via env vars or CLI flags:
    AUTOKERNEL_BENCH_CMD   shell command that prints "score: <float>" to stdout
    AUTOKERNEL_TEST_CMD    shell command for correctness (exit 0 = pass; "" = skip)
    AUTOKERNEL_TRIALS      how many bench runs to average (default 3)
    AUTOKERNEL_TIMEOUT_S   per-trial timeout (default 600)
    AUTOKERNEL_SIGMA_K     improvement must exceed K * combined_sigma (default 2.0)

Exit code: 0 if KEEP, 1 if REVERT, 2 if harness error.
"""
from __future__ import annotations
import argparse, json, os, re, statistics, subprocess, sys
from pathlib import Path

SCORE_RE = re.compile(r"score:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I)


def run_one(cmd: str, timeout: int) -> float:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"bench rc={p.returncode}: {p.stderr[-400:]}")
    m = SCORE_RE.search(p.stdout)
    if not m:
        raise RuntimeError(f"no 'score: <num>' in stdout: {p.stdout[-400:]}")
    return float(m.group(1))


def trials(cmd: str, n: int, timeout: int) -> dict:
    samples = [run_one(cmd, timeout) for _ in range(n)]
    mean = statistics.mean(samples)
    sd = statistics.stdev(samples) if n > 1 else 0.0
    return {
        "mean": mean,
        "stddev": sd,
        "sigma_pct": (100 * sd / mean) if mean else 0.0,
        "samples": samples,
        "n": n,
    }


def correctness(cmd: str, timeout: int) -> dict:
    if not cmd.strip():
        return {"ran": False, "passed": True, "log_tail": "(skipped)"}
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return {
        "ran": True,
        "passed": p.returncode == 0,
        "log_tail": (p.stdout + p.stderr)[-400:],
    }


def verdict(cur: dict, base: dict | None, sigma_k: float) -> dict:
    if not cur["correctness"]["passed"]:
        return {"keep": False, "reason": "correctness failed", "delta_pct": None}
    if base is None:
        return {"keep": True, "reason": "first run; becomes baseline", "delta_pct": None}
    cm, cs = cur["score"]["mean"], cur["score"]["stddev"]
    bm, bs = base["score"]["mean"], base["score"]["stddev"]
    if bm == 0:
        return {"keep": False, "reason": "baseline mean is zero", "delta_pct": None}
    delta_pct = 100.0 * (cm - bm) / bm
    combined_sd = (cs * cs + bs * bs) ** 0.5
    needed = sigma_k * combined_sd
    if cm - bm < needed:
        return {
            "keep": False,
            "reason": f"delta {delta_pct:+.2f}% inside {sigma_k}*sigma noise floor "
                      f"(need ≥ {100*needed/bm:+.2f}%)",
            "delta_pct": delta_pct,
        }
    return {
        "keep": True,
        "reason": f"improved {delta_pct:+.2f}% past noise floor",
        "delta_pct": delta_pct,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench-cmd", default=os.environ.get("AUTOKERNEL_BENCH_CMD"))
    ap.add_argument("--test-cmd", default=os.environ.get("AUTOKERNEL_TEST_CMD", ""))
    ap.add_argument("--trials", type=int, default=int(os.environ.get("AUTOKERNEL_TRIALS", 3)))
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("AUTOKERNEL_TIMEOUT_S", 600)))
    ap.add_argument("--sigma-k", type=float, default=float(os.environ.get("AUTOKERNEL_SIGMA_K", 2.0)))
    ap.add_argument("--baseline", type=Path, default=Path("baseline.json"),
                    help="JSON of last-good run; absent = this run becomes baseline")
    ap.add_argument("--out", type=Path, default=Path("current.json"))
    args = ap.parse_args()

    if not args.bench_cmd:
        sys.exit("error: set --bench-cmd or AUTOKERNEL_BENCH_CMD")

    try:
        score = trials(args.bench_cmd, args.trials, args.timeout)
        ok = correctness(args.test_cmd, args.timeout)
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(2)

    cur = {"score": score, "correctness": ok}
    base = json.loads(args.baseline.read_text()) if args.baseline.exists() else None
    v = verdict(cur, base, args.sigma_k)
    out = {**cur, **v}
    args.out.write_text(json.dumps(out, indent=2))
    if v["keep"]:
        args.baseline.write_text(json.dumps(cur, indent=2))
    print(json.dumps(out, indent=2))
    sys.exit(0 if v["keep"] else 1)


if __name__ == "__main__":
    main()
