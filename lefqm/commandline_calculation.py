import tempfile
from pathlib import Path


class CommandlineCalculation:
    """Commandline calculation"""

    def __init__(self, config, run_dir_path=None):
        """Commandline calculation

        :param config: config for the calculation
        :type config: dict
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
