"""Commandline calculations"""
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from lefqm.conformator import conformator_generate
from lefqm.gaussian import gaussian_calculate_shieldings
from lefqm.nwchem import nwchem_calculate_shieldings
from lefqm.omega import omega_generate
from lefqm.turbomole import turbomole_calculate_shieldings


class CommandlineCalculation:
    """Commandline calculation"""

    def __init__(self, config, run_dir_path=None):
        """Commandline calculation

        :param config: config for the calculation
        :type config: configparser.ConfigParser
        :param run_dir_path: path to the directory to run in
        :type run_dir_path: pathlib.Path
        """
        self.config = config
        self.run_dir_path = run_dir_path
        self.tmp_dir = None
        if self.run_dir_path is None:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.run_dir_path = Path(self.tmp_dir.name)

    def __del__(self):
        """Delete commandline calculation

        Remove the temporary directory if one was created.
        """
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()


class ConformerGeneration(CommandlineCalculation):
    """Conformer generation"""

    def run(self, mol):
        """Run conformer generation

        :param mol: molecule to run the generation for
        :type mol: rdkit.Chem.rdchem.Mol
        :return: mol with conformations
        :rtype: rdkit.Chem.rdchem.Mol
        """
        if self.config["Workflow"]["confgen_method"] == "conformator":
            return conformator_generate(
                mol,
                conformator=self.config["Paths"]["conformator"],
                max_confs=int(self.config["Parameters"]["max_confs"]),
                run_dir_path=self.run_dir_path,
            )
        if self.config["Workflow"]["confgen_method"] == "rdkit":
            mol = Chem.AddHs(mol)
            AllChem.EmbedMultipleConfs(mol, numConfs=int(self.config["Parameters"]["max_confs"]))
            return mol
        if self.config["Workflow"]["confgen_method"] == "omega":
            return omega_generate(
                mol,
                omega=self.config["Paths"]["omega"],
                max_confs=int(self.config["Parameters"]["max_confs"]),
                cores=int(self.config["Parameters"]["cores"])
                if self.config["Parameters"]["cores"] is not None
                else None,
                prune_threshold=float(self.config["Parameters"]["conf_prune_threshold"]),
            )
        raise RuntimeError("Invalid QM method")


class ShieldingCalculation(CommandlineCalculation):
    """NMR shielding constant calculation"""

    def run(self, mol):
        """Run shielding calculation

        :param mol: molecule to run the calcualtion on
        :type mol: rdkit.Chem.rdchem.Mol
        :return: shieldings for every atom
        :rtype: list[float]
        """
        if self.config["Workflow"]["qm_method"] == "turbomole":
            return turbomole_calculate_shieldings(
                mol,
                x2t=self.config["Paths"]["x2t"],
                ridft=self.config["Paths"]["ridft"],
                mpshift=self.config["Paths"]["mpshift"],
                cores=self.config["Parameters"]["cores"],
                run_dir_path=self.run_dir_path,
            )
        if self.config["Workflow"]["qm_method"] == "nwchem":
            return nwchem_calculate_shieldings(
                mol,
                nwchem=self.config["Paths"]["nwchem"],
                run_dir_path=self.run_dir_path,
            )
        if self.config["Workflow"]["qm_method"] == "gaussian":
            return gaussian_calculate_shieldings(
                mol,
                gaussian=self.config["Paths"]["gaussian"],
                run_dir_path=self.run_dir_path,
            )
        raise RuntimeError("Invalid QM method")
