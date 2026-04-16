import subprocess
import threading
from pathlib import Path

class IlastikRunner :

    def __init__(self, ilastik_model: str, ilastik_exe: str) :
        self._ilastik_model = str(Path(ilastik_model).resolve())
        self._ilastik_exe = ilastik_exe

        if not Path(self._ilastik_model).exists():
            raise FileNotFoundError(f"ilastik model not found: {self._ilastik_model}")
        if not Path(self._ilastik_exe).exists():
            raise FileNotFoundError(
                f"ilastik executable not found: {self._ilastik_exe}\n"
                "Set 'ilastik_exe' in configuration"
            )

    def run_on_frames(self, input_paths: list, output_dir: str, n_channels: int = 3) :
        axes = "yxc" if n_channels > 1 else "yx"
        self._run_headless(input_paths, output_dir, axes)


    #internal method, run and handle new process
    def _run_headless(self, input_paths: list, output_dir: str, axes: str) :
        cmd = [
            self._ilastik_exe,
            "--headless",
            f"--project={self._ilastik_model}",
            "--export_source=Probabilities",
            "--output_format=hdf5",
            f"--input_axes={axes}",
            f"--output_filename_format={output_dir}/{{nickname}}_{{result_type}}.h5",
        ] + input_paths

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        _, stderr = process.communicate()

        if process.returncode != 0 :
            raise RuntimeError(
                f"ilastik headless failed (exit {process.returncode}):\n"
                + stderr.decode(errors="replace")
            )


