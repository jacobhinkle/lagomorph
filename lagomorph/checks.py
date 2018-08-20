from pycuda import gpuarray
from pycuda.reduction import ReductionKernel
from pycuda.tools import dtype_to_ctype
import numpy as np

class ContextCheck:
    def __init__(self, constructor, *args, check_nans=False, check_infs=False, **kwargs):
        self.constructor = constructor
        self.args = args
        self.kwargs = kwargs
        self.check_nans = check_nans
        self.check_infs = check_infs
        self.f = None
    def check_context(self):
        if self.f is None:
            self.f = self.constructor(*self.args, **self.kwargs)
    def __call__(self, *args, **kwargs):
        self.check_context()
        for a in args:
            if self.check_nans:
                if isinstance(a, gpuarray.GPUArray) and \
                        a.dtype in [np.float32, np.float64]:
                    nn = count_nans(a)
                    if nn > 0:
                        raise Exception(f"Found {nn} nans")
            if self.check_infs:
                if isinstance(a, gpuarray.GPUArray) and \
                        a.dtype in [np.float32, np.float64]:
                    nn = count_infs(a)
                    if nn > 0:
                        raise Exception(f"Found {nn} infs")
        return self.f(*args, **kwargs)

_cnkernels = {xtype: ContextCheck(ReductionKernel,
                np.int64, neutral="0",
                check_nans=False,
                check_infs=False,
                reduce_expr="a+(b!=b ? 1 : 0)",
                arguments=f"const {xtype} *in") for xtype in
                ['float','double']}
def count_nans(x):
    return np.isnan(x.get()).sum()
    xtype = dtype_to_ctype(x.dtype)
    return _cnkernels[xtype](x).get()

_cikernels = {xtype: ContextCheck(ReductionKernel,
                np.int64, neutral="0",
                check_nans=False,
                check_infs=False,
                reduce_expr="a+((b==INFINITY || b==-INFINITY) ? 1 : 0)",
                arguments=f"const {xtype} *in") for xtype in
                ['float','double']}
def count_infs(x):
    return np.isinf(x.get()).sum()
    xtype = dtype_to_ctype(x.dtype)
    return _cikernels[xtype](x).get()
