

import os
from pathlib import Path
import shutil

path = "/workspaces/TimeSFormer/timesformer/datasets/spread/2"

pattern = "image_{}"

file_names = os.listdir(path)
for file_name in file_names:
    suffix = file_name.split("_")[-1]
    file_path = Path(file_name)
    new_file_name = pattern.format(suffix)
    new_file_location = Path(path).joinpath(new_file_name)
    old_file_location = Path(path).joinpath(file_name)
    shutil.move(old_file_location.as_posix(),new_file_location.as_posix())
