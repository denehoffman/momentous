# Releasing momentous

Workflows are generated from [`.yamloom.py`](../.yamloom.py). Edit that file,
run `uv run yamloom sync`, and include the generated YAML with your changes.
`uv run yamloom check` verifies that the files are current without editing them.

## One-time setup

1. Add a GitHub repository secret named `RELEASE_PLEASE`, following the same
   convention as laddu and pdg-rs. Use a token with permission to create release
   PRs, labels, tags, and GitHub releases in this repository. A personal access
   token allows those PRs and tags to trigger CI and publishing; the default
   `GITHUB_TOKEN` would suppress those subsequent workflow runs.
2. Create the GitHub Actions environment `pypi`. Configure any required reviewer
   and release-tag restrictions there.
3. On PyPI, configure a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
   for project `momentous`, using the GitHub owner and repository hosting this
   checkout, workflow filename `publish.yml`, and environment name `pypi`.
   Publishing uses GitHub OIDC; no PyPI API token belongs in the workflow.

These are account settings, not files that yamloom can configure.

## First release and version policy

The empty [release manifest](../.release-please-manifest.json) means that this
package has never been released. The Python strategy starts at **0.1.0**, matching
`pyproject.toml`. Do not seed the manifest with `0.1.0`: it records completed
releases, not the next planned version.

The extra-file JSONPath selects momentous by name in release-please's tagged
TOML representation (`name.value`). Its updater preserves the lockfile's
formatting and leaves dependency versions unchanged.

The [release configuration](../release-please-config.json) enables
`bump-minor-pre-major` and `bump-patch-for-minor-pre-major`. Below 1.0:

| Commit | Example next version from 0.1.0 |
| --- | --- |
| `fix: ...` | 0.1.1 |
| `feat: ...` | 0.1.1 |
| `feat!: ...` or a `BREAKING CHANGE` footer | 0.2.0 |

These options follow the [release-please configuration](https://github.com/googleapis/release-please/blob/main/docs/manifest-releaser.md).
They do not automatically advance the package to 1.0.0. Plan that release
explicitly when the interface is ready.

## Release sequence

1. Merge conventional commits into `main`. Release Please opens or updates a
   release PR with the changelog, `pyproject.toml`, the package's version entry
   in `uv.lock`, and the release manifest.
2. Review the proposed version and notes, then merge the release PR once CI
   passes. Release Please creates the `vVERSION` tag and GitHub release.
3. The tag starts `publish.yml`. It reruns the code checks, verifies that the tag
   matches the package version, builds a source distribution, builds the wheel
   from that distribution, and tests the installed wheel and its docstrings on
   Python 3.12, 3.13, and 3.14.
4. After all checks pass and any `pypi` environment approval is granted, the
   publish job downloads the tested distributions and uploads them to PyPI.

No publishing occurs on an ordinary branch push or pull request. Rerun a failed
tag workflow after fixing account configuration; published versions cannot be
replaced.

## Local checks

```sh
uv sync --locked
uv run prek install
uv run prek run --all-files
uv run yamloom check
uv build
```

Ruff, ty, pytest, prek, and yamloom are pinned in the development group. The
hooks use `uv run --locked`, so local checks and CI use the same tool versions.
