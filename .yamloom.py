"""Generate the CI, release-please, and PyPI publication workflows.

Run ``uv run yamloom sync`` after editing this file. Generated YAML belongs in
version control; ``uv run yamloom check`` verifies that it is current.
"""

from yamloom import (
    Concurrency,
    Environment,
    Events,
    Job,
    Matrix,
    Permissions,
    PullRequestEvent,
    PushEvent,
    Strategy,
    Workflow,
    script,
    sync,
)
from yamloom.actions.github.artifacts import DownloadArtifact, UploadArtifact
from yamloom.actions.github.release import ReleasePlease
from yamloom.actions.github.scm import Checkout
from yamloom.actions.packaging.python import PypiPublish
from yamloom.actions.toolchains.python import SetupUV
from yamloom.expressions import context

READ = Permissions(contents="read")

CHECKS = Job(
    name="Lint, types, and workflows",
    runs_on="ubuntu-latest",
    timeout_minutes=10,
    steps=[
        Checkout(),
        SetupUV(python_version="3.12", enable_cache=True),
        script("uv sync --locked"),
        # The wheel matrix runs pytest, including all docstring examples.
        script("uv run --locked prek run --all-files", env={"SKIP": "pytest"}),
        script("uv run --locked yamloom check"),
        script("git diff --exit-code"),
    ],
)

WHEEL_TESTS = Job(
    name="Installed wheel / Python ${{ matrix.python }}",
    runs_on="ubuntu-latest",
    timeout_minutes=20,
    strategy=Strategy(matrix=Matrix(python=["3.12", "3.13", "3.14"]), fast_fail=False),
    steps=[
        Checkout(),
        SetupUV(python_version="${{ matrix.python }}", enable_cache=True),
        script("uv sync --locked --no-install-project"),
        script(
            """uv run --no-sync python - <<'PY'
import os
import tomllib

with open("pyproject.toml", "rb") as source:
    version = tomllib.load(source)["project"]["version"]
if os.environ["GITHUB_REF_NAME"] != f"v{version}":
    raise SystemExit("Release tag must match the version in pyproject.toml")
PY""",
            name="Verify release version",
            condition=context.github.ref_type == "tag",
        ),
        script(
            "uv build --sdist",
            "uv build dist/*.tar.gz --wheel --out-dir dist",
            "uv pip install --no-deps dist/*.whl",
            name="Build the wheel from the source distribution",
        ),
        script(
            """uv run --no-sync python - <<'PY'
from pathlib import Path

import momentous
import pytest

package = Path(momentous.__file__).resolve().parent
if package.is_relative_to(Path("src").resolve()):
    raise SystemExit("Tests must import the installed wheel")
raise SystemExit(pytest.main(["tests", str(package)]))
PY""",
            name="Test the installed wheel and its docstrings",
        ),
        UploadArtifact(
            path="dist/*",
            artifact_name="distributions",
            condition="matrix.python == '3.12'",
        ),
    ],
)

ci = Workflow(
    name="CI",
    on=Events(push=PushEvent(branches=["main"]), pull_request=PullRequestEvent()),
    permissions=READ,
    concurrency=Concurrency(
        "${{ github.workflow }}-${{ github.ref }}", cancel_in_progress=True
    ),
    jobs={"checks": CHECKS, "tests": WHEEL_TESTS},
)

release_please = Workflow(
    name="Release Please",
    on=Events(push=PushEvent(branches=["main"])),
    permissions=READ,
    concurrency=Concurrency("release-please-main", cancel_in_progress=False),
    jobs={
        "release-please": Job(
            runs_on="ubuntu-latest",
            timeout_minutes=10,
            steps=[ReleasePlease(token=context.secrets.RELEASE_PLEASE)],
        )
    },
)

publish = Workflow(
    name="Publish to PyPI",
    on=Events(push=PushEvent(tags=["v*"])),
    permissions=READ,
    concurrency=Concurrency("publish-${{ github.ref }}", cancel_in_progress=False),
    jobs={
        "checks": CHECKS,
        "tests": WHEEL_TESTS,
        "publish": Job(
            runs_on="ubuntu-latest",
            needs=["checks", "tests"],
            environment=Environment("pypi"),
            permissions=Permissions(contents="read", id_token="write"),
            timeout_minutes=10,
            steps=[
                DownloadArtifact(artifact_name="distributions", path="dist"),
                PypiPublish(),
            ],
        ),
    },
)

if __name__ == "__main__":
    sync({"ci.yml": ci, "release-please.yml": release_please, "publish.yml": publish})
