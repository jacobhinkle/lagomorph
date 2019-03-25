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
        group.add_argument('--world_size', default=1, help='Number of NCCL ranks')
        group.add_argument('--rank', default=0, help='NCCL rank')
        group.add_argument('--gpu', default="local_rank", type=str, help='GPU to use, None for CPU, "local_rank" to use value of --local_rank')
        group.add_argument('--local_rank', default=0, type=int, help='Local NCCL rank')
    def _stamp_dataset(self, ds, args):
        from .version import __version__
        import json
        ds.attrs['lagomorph_version'] = __version__
        ds.attrs['command_args'] = json.dumps(vars(args))
