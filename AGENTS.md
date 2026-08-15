# AGENTS.md

LEFQM: Python CLI (`lefqm`) that predicts 19F chemical shifts via a split QM workflow. Four subtools, each a module under `lefqm/` with a matching `add_*_subparser` + entrypoint wired in `lefqm/__main__.py`:

1. `conformers` -> SMILES/CSV -> one SD per molecule
2. `shieldings` -> SD -> SD with shielding constants as atom double-property list
3. `ensembles` -> SDFs -> shieldings CSV (CF/CF2/CF3 averaged)
4. `shifts` -> shieldings CSV -> chemical shifts via linear regression against a calibration CSV

## Commands

- Tests (unittest, not pytest): `python -m unittest tests`; run one file with `python -m unittest tests.utils_tests`. Tests must be run from the repo root (they use relative paths like `tests/data/...`).
- Formatting: pre-commit only (`autoflake`, `black --line-length=99`, `flake8`). `pre-commit run --files`. Code is black-formatted at 99 cols.
- Lint: `pylint lefqm tests` (must stay >9.0). `.pylintrc` ignores rdkit/lef/lefqm.constants.
- Coverage: `coverage run --source=lefqm -m unittest tests && coverage html` (thresholds: 90% overall, 80% per file).

## Config & architecture

- Subtools default to `lefqm/config.ini`; a `--config` file is merged over it (`utils.get_config`, configparser semantics). `[Paths]` entries are the exact shell commands invoked. QM engine is chosen by `Workflow.qm_method` (turbomole|nwchem|gaussian); conformer generator by `Workflow.confgen_method` (conformator|rdkit|omega); moka protonation is opt-in via `Parameters.protonate`.
- The repo-root `config.ini` is an untracked local override in this environment, not part of the repo.
- `lefqm` re-exports constants from `lefshift` (`lefqm/constants.py`), e.g. `SHIELDING_SD_PROPERTY`, column names. Cross-package contract: `lefshift` must stay importable.
- External tool wrappers live in `lefqm/{conformator,omega,xtb,moka,turbomole,nwchem,gaussian}.py`; dispatch via `lefqm/commandline_calculation.py`.

## Testing gotchas

- Tests are NOT hermetic. `tests/commandline_calculation_tests.py` and `tests/main_tests.py` shell out to real QM binaries (`conformator`, `xtb`, `nwchem`, plus licensed/missing `turbomole` x2t/ridft/mpshift, `gaussian` g16, `omega`, `moka`/`blabber_sd`). In this dev env only `conformator`, `xtb`, and `nwchem` are on PATH, so the omega/turbomole/gaussian/moka tests fail for environmental reasons — not because of your change.
- `lefshift` is a hard dependency but is not pip-installed in this env; it is imported from the sibling checkout `/home/broker/projects/lefshift` (already on `sys.path`). `sklearn` (used by `shifts.py`) and `scipy` (used by `conformers.py`) are likewise undeclared in `pyproject.toml` — they arrive transitively via `lefshift`. Do not prune dependencies blindly.
- SDF output is forced V3000 with 10-decimal precision via `utils.generate_high_precision_sdmol`. `tests/data/high_precision_mol.sdf` is an exact-string snapshot: changing the format or precision breaks `test_write_high_precision_mol`.
- `conformers.normalize()` raises `RuntimeError` for molecules with unassigned diastereomers. `conformers`/`shieldings` catch exceptions per molecule and only log a warning, so failures on individual molecules don't abort the run.
