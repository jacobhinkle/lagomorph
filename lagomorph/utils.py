import argparse, sys

# from https://github.com/slundberg/shap/issues/121
IN_IPYNB = None

def in_ipynb():
  global IN_IPYNB
  if IN_IPYNB is not None:
    return IN_IPYNB

  try:
    cfg = get_ipython().config
    if type(get_ipython()).__module__.startswith("ipykernel."):
    # if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
      # print ('Running in ipython notebook env.')
      IN_IPYNB = True
      return True
    else:
      return False
  except NameError:
    # print ('NOT Running in ipython notebook env.')
    return False

if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

def mpi_size():
    try:
        from mpi4py import MPI
    except ImportError:
        return 1
    return MPI.COMM_WORLD.Get_size()

def mpi_rank():
    try:
        from mpi4py import MPI
    except ImportError:
        return 0
    return MPI.COMM_WORLD.Get_rank()

def mpi_local_comm():
    """Split MPI communicator based on shmem capability to produce a node-local
    communicator.

    Requires MPI implementation version >= 3
    """
    from mpi4py import MPI
    return MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)

def mpi_local_rank():
    """Computes the local rank using an MPI split communicator with type shmem.

    This is the same method horovod uses to decide local rank. This
    functionality is provided here in order to remain independent of horovod (in
    order to use hvd.local_rank() you need to initialize and use horovod).
    However, instead of using a custom C extension as hvd does, we use mpi4py
    which now supports these split communicators.

    Requires MPI implementation version >= 3
    """
    try:
        return mpi_local_comm().Get_rank()
    except NotImplementedError:
        # MPI < 3, fall back to naive hostname-based method
        print("Local rank computation not yet implemented for MPI < 3")
        return NotImplemented

class Tool:
    # Customized from:
    # https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html#
    module_name = None
    subcommands = []
    def __init__(self):
        usage = f"python -m {self.module_name} <command> [<args>]" \
            + "\n\nAvailable subcommands:\n\n"
        for c in self.subcommands:
            usage += f"{c:15s} {self.describe_subcommand(c)}\n"
        usage += "\n"
        parser = self.new_parser(usage=usage)
        parser.add_argument("command", help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if args.command not in self.subcommands:
            print('ERROR: Unrecognized command')
            parser.print_help()
            sys.exit(1)
        self.call_subcommand(args.command)
    def describe_subcommand(self, sub):
        return getattr(self, sub).__doc__
    def new_parser(self, subcmd=None, **kwargs):
        prog = 'python -m ' + self.module_name
        if subcmd is not None:
            prog += ' ' + subcmd
        return argparse.ArgumentParser(
                prog=prog,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                **kwargs)
    def call_subcommand(self, command):
        # use dispatch pattern to invoke method with same name
        getattr(self, command)()
    @staticmethod
    def _compute_args(parser):
        """Add common arguments for parallel commands"""
        group = parser.add_argument_group('compute parameters')
        group.add_argument('--gpu', default="local_rank", type=str, help='GPU to use, None for CPU, "local_rank" to use local MPI rank')
    def _initialize_compute(self, args):
        """Use common compute_args to initialize torch and NCCL"""
        self.rank = mpi_rank()
        self.world_size = mpi_size()
        self.local_rank = mpi_local_rank()

        self.gpu = args.gpu
        if self.gpu == 'local_rank':
            self.gpu = self.local_rank
        else:
            self.gpu = int(self.gpu)

        import torch
        torch.cuda.set_device(self.gpu)

        if self.world_size > 1:
            torch.distributed.init_process_group(backend='nccl',
                    world_size=self.world_size, rank=self.rank,
                    init_method='env://')
    def _stamp_dataset(self, ds, args):
        from .version import __version__
        import json
        ds.attrs['lagomorph_version'] = __version__
        ds.attrs['command_args'] = json.dumps(vars(args))
