import numpy as np
import utils


def read_bv(fn):
    with open(fn, 'rb') as f:
        model = utils.binvox_rw.read_as_3d_array(f)
    data = np.float32(model.data)
    return data
