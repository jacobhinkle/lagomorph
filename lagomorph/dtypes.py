import numpy as np

def dtype2precision(dt):
    if dt == np.float32:
        return 'single'
    elif dt == np.float64:
        return 'double'
    else:
        raise Exception(f"Unknown dtype {dt}")
