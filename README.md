# Local Environment of Fluorine (LEF) QM Tools

## Getting Started

A PyPi package is available:
```
pip install lefqm
```

You can also clone the repo manually and then install the lefqm package:
```
git clone https://github.com/PatrickPenner/lefqm.git
cd lefqm
pip install .
```

Running `lefqm --help` should give you an overview of how to use the lefqm
commandline tool.

To generate conformers for SMILES in a CSV file run:
```
mkdir -p data/conformers/
lefqm conformers data/test.csv data/conformers/ --id-column 'Catalog ID' --chunk 0 --chunk-size 1
```
Molecules should have a name for conformer generation. We specify a name/ID
column explicitly in the command above. The `conformers` subtool also supports
chunking, which is convenient for cluster execution. The above runs the zero-th
cluster of size one, i.e. the first molecule. The conformers tool will produce
one SD file containing all conformers for one molecule.

To calculate shielding constants for conformers run:
```
mkdir -p data/shieldings/
lefqm shieldings data/conformers/Z1672161820.sdf data/shieldings/Z1672161820.sdf
```
The new SD file will contain the shielding constants as an SD atom property list.

Shieldings for a conformer ensemble can be combined using:
```
lefqm ensembles data/shieldings/Z1672161820.sdf data/Z1672161820.csv
```
... or ...
```
lefqm ensembles data/shieldings/ data/shieldings.csv
```
... to combine all ensembles in a directory into one shieldings CSV.


Shieldings can be converted to shift with a calibration data set in the
following way:
```
sed -i 's/ID,/Catalog ID,/' data/shieldings.csv  # rename the ID column to make it consistent with the input data
lefqm shifts data/shieldings.csv --calibration data/train.csv data/shifts.csv --id-column 'Catalog ID' --shift-column 'Shift 1 (ppm)'
```
We use `train.csv` as a calibration set. A calibration set must contain a
shieldings constants column as well as a chemical shift column to train the
linear regression conversion from shieldings constants to chemical shifts.

## External dependencies

This package is written as a collection of QM workflow tools and therefore
relies on other commandline tools to perform calculations, specifically
these tools:

- [MoKa](https://www.moldiscovery.com/software/moka/) - protomerization/tautomerization
- [Conformator](https://www.zbh.uni-hamburg.de/forschung/amd/software/conformator.html) - conformer generation
- [Omega](https://www.eyesopen.com/) - conformer generation
- [xTB](https://xtb-docs.readthedocs.io/en/latest/contents.html) - conformer optimization
- [Turbomole binaries: x2t, ridft, mpshift](https://www.turbomole.org/) - DFT shielding constant calculations
- [NWChem](https://nwchemgit.github.io/) - DFT shielding constant calculations
- [Gaussian](https://gaussian.com/) - DFT shielding constant calculations

The above tools are redundant and you may choose between using, for example,
one of the three QM engines listed above. To configure which tools are supposed
to run you must pass a config file with `--config` to each commandline call.
You do not need to specify all parameters, but can only include those you want
to change. The default configuration can be found in `lefqm/config.ini` and
looks like this:

```ini
[Paths]
moka = blabber_sd
xtb = xtb
conformator = conformator
x2t = x2t
ridft = ridft
mpshift = mpshift
omega = omega2
nwchem = nwchem
gaussian = g16

[Workflow]
; Can be "conformator" / "rdkit" / "omega"
confgen_method = conformator
; Can be "turbomole" / "nwchem" / "gaussian"
qm_method = turbomole

[Parameters]
max_confs = 250
; In Angstroem RMSD
conf_prune_threshold = 0.15
```

Note the `Paths` section in the config file. The strings in that section will
be the exact commands lefqm will try to run. If these are different in your
environment, you will need to edit them.

Conformer generation and shielding constant calculation are intentionally
split from the rest of the workflow. If you do not have the tools available for
one of these steps or have different tool preferences you are welcome to leave
and re-enter the workflow at any point.

The output from the conformer step should be one SD file per input molecule
that contains all conformers that should be considered for that molecule. Be
aware that these geometries will go directly into a QM calculation that may
fail if they are not optimized to a QM level. XTB is a great trade-off between
CPU time and geometric quality to achieve this. Single SD files or a directory
of SD files can then be processed by the `shieldings` subtool.

The output from the shielding constant calculation step should be one SD file
per molecule that contains all conformers associated with shielding constants
as an SD atom property list as written by RDKit. Given a molecule and a list of
shielding constants in the same order as the atoms in the molecule such a list
can be written like this:
```
from lefqm import constants
from rdkit import Chem

...

for atom, shielding in zip(mol.GetAtoms(), shielding_constants):
    atom.SetDoubleProp(constants.SHIELDING_SD_PROPERTY, shielding)
Chem.CreateAtomDoublePropertyList(mol, constants.SHIELDING_SD_PROPERTY)
```
The `lefqm` package has a constants file, which contains the defaults for
column or SD property names that `lefqm` tools expect. `SHIELDING_SD_PROPERTY`
is the default name for shielding constants in SD files. You may also
explicitly specify the shielding property name on the commandline going forward.
Such output can then be fed back into the workflow through the `ensembles`
subtool.

## Development

### Formatting pre-commit hook

Pre-commit is only used for consistent formatting. The core of that is the
black formatter and code style.

Install the formatting pre-commit hook with:
```
pre-commit install
```

### Code Quality

All quality criteria have a range of leniency.

| Criteria               | Threshold     |
| -------------          |:-------------:|
| pylint                 | \>9.0         |
| coverage (overall)     | \>90%         |
| coverage (single file) | \>80%         |

### Utility commands

Pre-commit on one file:
```
pre-commit run --files
```

Test command:

```
python -m unittest tests
```

PyLint command:

```
pylint lefqm tests > pylint.out
```

Coverage commands:
```
coverage run --source=lefqm -m unittest tests
coverage html
```
