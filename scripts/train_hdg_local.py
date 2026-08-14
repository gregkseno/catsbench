#!/usr/bin/env python3
"""Run one HD benchmark method locally, four experiments at a time."""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


METHODS = ("dlight_sb", "dlight_sb_m", "csbm", "alpha_csbm", "c2sbm")
DIMS = (2, 16, 64)
BENCHMARKS = ("g002", "g005", "u0005", "u001")
VARIANTS = ("t63_kl", "t63_mse", "t15_kl", "t15_mse")


def experiments(method: str) -> list[str]:
    if method == "dlight_sb":
        return [
            f"{method}/benchmark_hd/d{dim}_{benchmark}"
            for dim in DIMS
            for benchmark in BENCHMARKS
        ]
    return [
        f"{method}/benchmark_hd/d{dim}_{benchmark}_{variant}"
        for dim in DIMS
        for benchmark in BENCHMARKS
        for variant in VARIANTS
    ]


def train(experiment: str, seed: int) -> None:
    print(f"START {experiment}", flush=True)
    command = [
        sys.executable,
        "-m",
        "src.run",
        f"experiment={experiment}",
        f"seed={seed}",
        "data.num_workers=0",
        "data.pin_memory=false",
        "logger=csv",
        "ckpt_path=null",
    ]
    subprocess.run(command, check=True)
    print(f"DONE  {experiment}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        jobs = {
            executor.submit(train, experiment, args.seed): experiment
            for experiment in experiments(args.method)
        }
        for job in as_completed(jobs):
            experiment = jobs[job]
            try:
                job.result()
            except Exception as error:
                failures.append(experiment)
                print(f"FAILED {experiment}: {error}", file=sys.stderr, flush=True)

    if failures:
        raise SystemExit(f"{len(failures)} experiment(s) failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
