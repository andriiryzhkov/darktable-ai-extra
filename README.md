# Darktable AI Extra

A community model repository for darktable: models that are useful but sit
outside the reference catalogue, packaged as `.dtmodel` archives with the
[darktable-ai-sdk](https://github.com/darktable-org/darktable-ai-sdk).

> **Not an official darktable project.** This repository is not maintained,
> supported or endorsed by the darktable project, and nothing here is
> reviewed by it. Do not report issues with these models to darktable –
> open them here instead.

Models land here for reasons that keep them out of
[darktable-ai](https://github.com/darktable-org/darktable-ai): a licence that
doesn't suit a default install, a task darktable has no built-in support for,
an experimental export, or simply that nobody has proposed them upstream.
That makes this a place to try things, not a stable base to build on.

**Check the licence before you rely on a model.** Each `model.yaml` carries a
`model_card` with the licence of the weights *and* of the data they were
trained on, and those often differ – permissively licensed weights trained on
research-only data are common. The terms are yours to comply with; packaging a
model here is not a judgement that your use of it is permitted.

## Layout

- `models/<id>/model.yaml` – one directory per model
- `samples/` – sample inputs used by demo / validation steps
- `vendor/` – vendored third-party code and submodules
- `output/` – build artifacts (gitignored)

## Usage

`dtai` comes from the SDK and finds this repo by walking up from the current
directory – there's nothing to wire up:

```console
dtai list                # show available models
dtai setup <id>          # fetch checkpoints
dtai convert <id>        # export ONNX + config.json
dtai validate <id>       # sanity-check the export
dtai package <id>        # emit output/<id>.dtmodel
dtai run <id>            # do all of the above
```

Omit `<id>` to iterate over every model in `models/`.

## Continuous integration

`.github/workflows/` ships three workflows, all driven by
`dtai list --json-output`, which emits one matrix entry per non-skipped
model so each builds in its own job with only its own `dep_group`
installed:

- **check-pr.yml** – on every pull request: setup, convert, validate each
  model.
- **nightly.yml** – when a model definition changes on the default branch:
  build everything and roll a single `nightly` pre-release, so testers get
  updated weights without waiting for a release.
- **release.yml** – on a `release-*` tag: the same, then package, and
  attach every `.dtmodel` plus `versions.json` to a **draft** GitHub
  release for you to review and publish.

All three need `contents: write` for the publishing steps, which is
declared in the workflows themselves – no repository settings to change
beyond allowing Actions to write, if your org restricts that by default.

## Releasing

1. Bump `version:` in the `model.yaml` of whatever changed. darktable
   compares that string to decide an update is available, so an unchanged
   version means users keep the old weights.
2. Tag `release-<version>` and push it.
3. Review the draft release and publish.

`releases-index.json` at the repo root maps a darktable version to the
release tag it should pull from. It starts empty, which means "offer
nothing" – fill it in once you have published a release:

```json
{"schema": 1, "compatible_releases": {"5.6.0": "release-1.0.0"}}
```

Installing a `.dtmodel` by hand needs none of this: darktable's
preferences has an *install from file* action. See the SDK's
`docs/dtmodel-format.md` for how the download path works end to end.

## Extending the CLI

To add repo-specific commands (demo, benchmark, release, ...) or a
dependency-sync hook for models with conflicting dependencies, write them in
`darktable_ai_extra/cli.py` and uncomment the `[tool.dtai]` block in
`pyproject.toml`. `dtai` imports that module on first command lookup.

Don't declare a `dtai` console script of your own – two packages owning the
same script name collide, and install order picks the winner.
