#!/usr/bin/env python3
"""Run Rust parity suites and generate a Markdown report."""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


TEST_RESULT = re.compile(r"^test (.+) \.\.\. (ok|FAILED|ignored)$", re.MULTILINE)
METRIC_MARKER = "PARITY_METRIC\t"


@dataclass(frozen=True)
class Suite:
    name: str
    command: tuple[str, ...]
    minimum_tests: int
    metric_paths: frozenset[str]


@dataclass(frozen=True)
class Metric:
    path: str
    reference: str
    error: float
    budget: float


@dataclass
class Result:
    suite: Suite
    returncode: int
    elapsed: float
    tests: list[tuple[str, str]]
    metrics: list[Metric]
    output: str
    error: str | None = None

    @property
    def issues(self) -> list[str]:
        issues = []
        if self.error:
            issues.append(self.error)
        if self.returncode:
            issues.append(f"cargo exited with status {self.returncode}")
        passed = sum(status == "ok" for _, status in self.tests)
        if passed < self.suite.minimum_tests:
            issues.append(f"expected at least {self.suite.minimum_tests} passing tests, found {passed}")
        paths = {metric.path for metric in self.metrics}
        missing = sorted(self.suite.metric_paths - paths)
        if missing:
            issues.append(f"missing metrics: {', '.join(missing)}")
        return issues

    @property
    def passed(self) -> bool:
        return not self.issues


def suites() -> dict[str, Suite]:
    test_args = ("--", "--show-output", "--test-threads=1")
    return {
        "ndarray": Suite("NdArray", ("cargo", "test", "parity") +
            test_args, 10, frozenset({"NdArray f32", "NdArray portable W8",
                "NdArray portable W4 (block 8)"})),
        "wgpu": Suite("WGPU", ("cargo", "test", "--features", "wgpu",
            "gpt::parity::test_f16_w8_w4_logit_error_budgets") + test_args,
            1, frozenset({"WGPU f16", "WGPU native W8", "WGPU native W4 (block 8)"})),
    }


def parse_metrics(output: str) -> list[Metric]:
    metrics = []
    for line in output.splitlines():
        marker = line.find(METRIC_MARKER)
        if marker < 0:
            continue
        fields = line[marker:].split("\t")
        if len(fields) != 5:
            raise ValueError(f"invalid parity metric: {line}")
        _, path, reference, error, budget = fields
        metrics.append(Metric(path, reference, float(error), float(budget)))
    return metrics


def run_suite(root: Path, suite: Suite) -> Result:
    print(f"Running {suite.name} parity suite...", file=sys.stderr, flush=True)
    started = time.monotonic()
    env = {**os.environ, "CARGO_TERM_COLOR": "never"}
    try:
        process = subprocess.run(suite.command, cwd=root, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        output, returncode, error = process.stdout, process.returncode, None
    except OSError as exception:
        output, returncode, error = "", 127, str(exception)

    try:
        metrics = parse_metrics(output)
    except ValueError as exception:
        metrics, error = [], str(exception)
    tests = TEST_RESULT.findall(output)
    result = Result(suite, returncode, time.monotonic() - started, tests, metrics, output, error)
    print(f"{suite.name}: {'PASS' if result.passed else 'FAIL'} ({result.elapsed:.2f}s)",
        file=sys.stderr)
    return result


def git_revision(root: Path) -> str:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(("git", *args), cwd=root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)

    revision = git("rev-parse", "--short", "HEAD")
    if revision.returncode:
        return "unknown"
    dirty = bool(git("status", "--porcelain").stdout.strip())
    return revision.stdout.strip() + (" (dirty)" if dirty else "")


def report(results: list[Result], root: Path) -> str:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = ["# Parity Report", "", f"- Generated: `{generated}`",
        f"- Revision: `{git_revision(root)}`", f"- Host: `{platform.platform()}`", "",
        "## Summary", "", "| Suite | Result | Passed | Ignored | Duration |",
        "|---|---|---:|---:|---:|"]
    for result in results:
        passed = sum(status == "ok" for _, status in result.tests)
        ignored = sum(status == "ignored" for _, status in result.tests)
        lines.append(f"| {result.suite.name} | {'PASS' if result.passed else 'FAIL'} | "
            f"{passed} | {ignored} | {result.elapsed:.2f}s |")

    metrics = [metric for result in results for metric in result.metrics]
    lines += ["", "## Numerical Error Budgets", "",
        "| Path | Reference | Max absolute error | Budget | Result |",
        "|---|---|---:|---:|---|"]
    for metric in metrics:
        status = "PASS" if metric.error <= metric.budget else "FAIL"
        lines.append(f"| {metric.path} | {metric.reference} | {metric.error:.8g} | "
            f"{metric.budget:.8g} | {status} |")
    if not metrics:
        lines.append("| - | - | - | - | FAIL |")

    lines += ["", "## Checks", "", "| Suite | Test | Result |", "|---|---|---|"]
    for result in results:
        for name, status in sorted(result.tests):
            lines.append(f"| {result.suite.name} | `{name}` | {status.upper()} |")

    failures = [result for result in results if not result.passed]
    if failures:
        lines += ["", "## Failure Output"]
        for result in failures:
            details = "\n".join(f"- {issue}" for issue in result.issues)
            if result.output.strip():
                details += f"\n\n{result.output.strip()}"
            lines += ["", f"### {result.suite.name}", "", "```text", details, "```"]
    return "\n".join(lines) + "\n"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("all", "ndarray", "wgpu"), default="all",
        help="parity suite to run (default: all)")
    parser.add_argument("--output", type=Path, default=root / "target/parity-report.md",
        help="Markdown output path, or '-' for stdout")
    args = parser.parse_args()

    available = suites()
    selected = available.values() if args.backend == "all" else (available[args.backend],)
    results = [run_suite(root, suite) for suite in selected]
    markdown = report(results, root)
    if str(args.output) == "-":
        sys.stdout.write(markdown)
    else:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        print(f"Wrote parity report to {output}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
