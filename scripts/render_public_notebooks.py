#!/usr/bin/env python3
"""Execute and export the public notebooks for the demo repository."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NotebookSpec:
    slug: str
    path: Path
    execute_by_default: bool


def parse_args() -> argparse.Namespace:
    project2_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render public notebooks to HTML")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project2_root / "docs" / "rendered",
        help="Directory for rendered notebook artifacts.",
    )
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Export HTML without executing the runnable public notebooks first.",
    )
    parser.add_argument(
        "--execute-training",
        action="store_true",
        help="Execute the training notebook as well. Off by default because it is heavy.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional notebook slugs to render. Choices: 01, 02, 03.",
    )
    return parser.parse_args()


def resolve_project2_root() -> Path:
    return Path(__file__).resolve().parents[1]


def notebook_specs(root: Path) -> list[NotebookSpec]:
    return [
        NotebookSpec("01", root / "notebooks" / "01_eda_and_problem_setup.ipynb", True),
        NotebookSpec("02", root / "notebooks" / "02_training_pipeline_design.ipynb", False),
        NotebookSpec("03", root / "notebooks" / "03_inference_replay_demo.ipynb", True),
    ]


def jupyter_cmd() -> list[str]:
    exe = shutil.which("jupyter")
    if exe:
        return [exe]
    raise FileNotFoundError("Could not find `jupyter` on PATH. Install notebook dependencies first.")


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    args = parse_args()
    root = resolve_project2_root()
    output_dir = args.output_dir.resolve()
    tmp_dir = root / "tmp" / "rendered_notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    selected = notebook_specs(root)
    if args.only:
        allowed = set(args.only)
        selected = [spec for spec in selected if spec.slug in allowed]
        unknown = sorted(allowed - {spec.slug for spec in notebook_specs(root)})
        if unknown:
            raise ValueError(f"Unknown notebook slug(s): {unknown}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("IPYTHONDIR", str(root / "tmp" / "ipython"))
    env.setdefault("JUPYTER_CONFIG_DIR", str(root / "tmp" / "jupyter_config"))

    nbconvert = jupyter_cmd() + ["nbconvert"]

    for spec in selected:
        if not spec.path.exists():
            raise FileNotFoundError(f"Notebook not found: {spec.path}")

        execute = spec.execute_by_default and not args.skip_execution
        if spec.slug == "02" and args.execute_training:
            execute = True

        source_for_html = spec.path
        if execute:
            executed_name = f"{spec.path.stem}.executed.ipynb"
            run_cmd(
                nbconvert
                + [
                    "--to",
                    "notebook",
                    "--execute",
                    str(spec.path),
                    "--output",
                    executed_name,
                    "--output-dir",
                    str(tmp_dir),
                ],
                cwd=root,
                env=env,
            )
            source_for_html = tmp_dir / executed_name

        run_cmd(
            nbconvert
            + [
                "--to",
                "html",
                str(source_for_html),
                "--output",
                f"{spec.path.stem}.html",
                "--output-dir",
                str(output_dir),
            ],
            cwd=root,
            env=env,
        )

        if execute:
            rendered_ipynb = output_dir / f"{spec.path.stem}.executed.ipynb"
            shutil.copy2(source_for_html, rendered_ipynb)

    print(f"[OK] Rendered notebooks written to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise
