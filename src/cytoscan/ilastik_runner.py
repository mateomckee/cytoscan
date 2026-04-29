import subprocess
import os
import numpy as np
import h5py
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

    def read_prob_map(self, prob_path: str) -> np.ndarray :
        #if any preprocessing is done, undo it here
        return self._read_probabilities(prob_path)

    def _read_probabilities(self, prob_path: str) -> np.ndarray :
        #check that .h5 file exists
        if not os.path.exists(prob_path) :
            #optionally perform search for file before erroring out
            raise FileNotFoundError(
                f"ilastik did not produce: {prob_path}\n"
            )
        
        #read .h5 file
        with h5py.File(prob_path, "r") as f :
            key = list(f.keys())[0]
            data = f[key][()]

        data = data.squeeze()

        if data.ndim != 3 :
            raise ValueError(f"expected 3D probability map (H, W, classes), got shape {data.shape} instead")
        
        #cell probability channel
        return data[:, :, 0]


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

