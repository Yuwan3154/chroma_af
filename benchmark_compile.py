#!/usr/bin/env python3
"""
Benchmark Chroma + OpenFold pipeline with and without torch.compile.

Runs:
1. One test run per config to verify compilation succeeds (timeout 180s)
2. 16 timed runs per config
3. Reports mean, median, min, max runtime

Usage:
    conda run -n openfold_chroma python benchmark_compile.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure chroma_af is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Minimal run: 1 PDB, 1 exp, 1 iter
NUM_RUNS = 16
FIRST_RUN_TIMEOUT = 300  # seconds (180s compilation + buffer)
PER_RUN_TIMEOUT = 300    # seconds (~60s/run expected, 300s cap for 320-res protein)
PDB_LIST = "/home/ubuntu/ProteinEBM/protein_ebm/data/data_lists/test_decoys.txt"
BASE_OUTPUT = "/home/ubuntu/chroma_af/benchmark_results"


def run_pipeline(compile_chroma: bool, compile_openfold: bool, output_dir: str) -> float:
    """Run pipeline once, return elapsed seconds. Raises on timeout or failure."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_chroma_openfold.py"),
        "--pdb_list", PDB_LIST,
        "--output_dir", output_dir,
        "--num_experiments", "1",
        "--num_iterations", "1",
        "--limit_pdb", "1",
    ]
    if compile_chroma:
        cmd.append("--compile_chroma")
    if compile_openfold:
        cmd.append("--compile_openfold")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=PER_RUN_TIMEOUT,
    )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
        )
    return elapsed


def benchmark_config(
    name: str,
    compile_chroma: bool,
    compile_openfold: bool,
) -> dict:
    """Run benchmark for one config. Returns dict with runtimes and stats."""
    output_dir = os.path.join(BASE_OUTPUT, name.replace(" ", "_"))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Test run to verify compilation
    print(f"\n[{name}] Test run (timeout={FIRST_RUN_TIMEOUT}s)...")
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_chroma_openfold.py"),
            "--pdb_list", PDB_LIST,
            "--output_dir", output_dir,
            "--num_experiments", "1",
            "--num_iterations", "1",
            "--limit_pdb", "1",
        ]
        if compile_chroma:
            cmd.append("--compile_chroma")
        if compile_openfold:
            cmd.append("--compile_openfold")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=FIRST_RUN_TIMEOUT,
        )
        test_elapsed = time.perf_counter() - start

        if result.returncode != 0:
            print(f"  FAILED: exit {result.returncode}")
            print(f"  stderr: {result.stderr[-500:]}")
            return {
                "config": name,
                "compile_chroma": compile_chroma,
                "compile_openfold": compile_openfold,
                "compilation_ok": False,
                "error": result.stderr[-500:] if result.stderr else "unknown",
                "runtimes": [],
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
            }
        print(f"  OK ({test_elapsed:.1f}s)")
    except subprocess.TimeoutExpired as e:
        print(f"  TIMEOUT after {FIRST_RUN_TIMEOUT}s")
        return {
            "config": name,
            "compile_chroma": compile_chroma,
            "compile_openfold": compile_openfold,
            "compilation_ok": False,
            "error": "timeout",
            "runtimes": [],
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    # 2. 16 timed runs
    runtimes = []
    print(f"[{name}] Running {NUM_RUNS} timed runs (timeout={PER_RUN_TIMEOUT}s each)...")
    for i in range(NUM_RUNS):
        try:
            elapsed = run_pipeline(compile_chroma, compile_openfold, output_dir)
            runtimes.append(elapsed)
            print(f"  Run {i+1}/{NUM_RUNS}: {elapsed:.1f}s")
        except subprocess.TimeoutExpired:
            print(f"  Run {i+1}/{NUM_RUNS}: TIMEOUT")
            break
        except Exception as e:
            print(f"  Run {i+1}/{NUM_RUNS}: FAILED - {e}")
            break

    if not runtimes:
        return {
            "config": name,
            "compile_chroma": compile_chroma,
            "compile_openfold": compile_openfold,
            "compilation_ok": True,
            "runtimes": [],
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    runtimes_sorted = sorted(runtimes)
    n = len(runtimes)
    return {
        "config": name,
        "compile_chroma": compile_chroma,
        "compile_openfold": compile_openfold,
        "compilation_ok": True,
        "runtimes": runtimes,
        "mean": sum(runtimes) / n,
        "median": runtimes_sorted[n // 2] if n % 2 else (runtimes_sorted[n // 2 - 1] + runtimes_sorted[n // 2]) / 2,
        "min": min(runtimes),
        "max": max(runtimes),
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=16, help="Number of timed runs per config")
    parser.add_argument("--timeout-first", type=int, default=180, help="Timeout for first run (compilation)")
    parser.add_argument("--timeout-run", type=int, default=240, help="Timeout per run (60s for small proteins)")
    args = parser.parse_args()

    global NUM_RUNS, FIRST_RUN_TIMEOUT, PER_RUN_TIMEOUT
    NUM_RUNS = args.runs
    FIRST_RUN_TIMEOUT = args.timeout_first
    PER_RUN_TIMEOUT = args.timeout_run

    os.makedirs(BASE_OUTPUT, exist_ok=True)

    configs = [
        ("baseline (no compile)", False, False),
        ("compile_chroma only", True, False),
        ("compile_openfold only", False, True),
    ]

    results = []
    for name, cc, co in configs:
        r = benchmark_config(name, cc, co)
        results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n{r['config']}:")
        print(f"  Compilation OK: {r['compilation_ok']}")
        if r.get("error"):
            print(f"  Error: {r['error'][:200]}...")
        if r.get("runtimes"):
            print(f"  Runs: {r['n']}")
            print(f"  Mean:   {r['mean']:.2f}s")
            print(f"  Median: {r['median']:.2f}s")
            print(f"  Min:    {r['min']:.2f}s")
            print(f"  Max:    {r['max']:.2f}s")

    out_path = os.path.join(BASE_OUTPUT, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    summary_path = os.path.join(BASE_OUTPUT, "benchmark_summary.txt")
    with open(summary_path, "w") as f:
        f.write("BENCHMARK SUMMARY\n")
        f.write("=" * 60 + "\n")
        for r in results:
            f.write(f"\n{r['config']}:\n")
            f.write(f"  Compilation OK: {r['compilation_ok']}\n")
            if r.get("error"):
                f.write(f"  Error: {str(r['error'])[:300]}\n")
            if r.get("runtimes"):
                f.write(f"  Runs: {r['n']}\n")
                f.write(f"  Mean:   {r['mean']:.2f}s\n")
                f.write(f"  Median: {r['median']:.2f}s\n")
                f.write(f"  Min:    {r['min']:.2f}s\n")
                f.write(f"  Max:    {r['max']:.2f}s\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
