from . import affine
from . import data
from . import lddmm
from .utils import Tool

class LagomorphTool(Tool):
    """"""
    module_name = 'lagomorph'
    subcommands = ['affine','data','lddmm']
    def call_subcommand(self, command):
        import sys
        # remove subcommand arg before passing it down
        del sys.argv[1]
        return getattr(globals()[command], '_Tool')()
    def describe_subcommand(self, command):
        return getattr(globals()[command], '_Tool').__doc__

if __name__ == '__main__':
    LagomorphTool()
