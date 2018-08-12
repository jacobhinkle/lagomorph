import numpy as np

def dtype2precision(dt):
    if dt == np.float32:
        return 'single'
    elif dt == np.float64:
        return 'double'
    else:
        raise Exception(f"Unknown dtype {dt}")

def dtype2ctype(dt):
    if dt == np.float32:
        return 'float'
    elif dt == np.float64:
        return 'double'
    elif dt == np.int32:
        return 'int'
    else:
        raise Exception(f"Can't convert dtype {dt} to c type name")


