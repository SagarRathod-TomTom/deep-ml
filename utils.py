import os
import glob
import torch
from constants import RUN_DIR_NAME


def find_current_run_number(target_dir):
    files = glob.glob(os.path.join(target_dir, '{}*'.format(RUN_DIR_NAME)))

    if len(files) == 0:
        return 1

    run = 0
    for file in files:
        current = int(os.path.split(file)[1].split('_')[1])
        if current > run:
            run = current

    # Return new run number
    return run + 1


def binarize(output: torch.FloatTensor, threshold: float = 0.50):
    output[output >= threshold] = 255
    output[output < threshold] = 0
    return output.to(torch.uint8)


