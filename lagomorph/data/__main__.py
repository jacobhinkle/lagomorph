from .__init__ import *

class DataCmdLine():
    # Customized from:
    # https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html#
    def __init__(self):
        import argparse, sys
        subcmds = ['downscale']
        subhelps = '\n'.join([f'  {c}: {getattr(self,c).__doc__}' for c in subcmds])
        parser = argparse.ArgumentParser(
                usage=f"python -m lagomorph.data <command> [<args>]\n\nAvailable subcommands:\n{subhelps}")
        parser.add_argument("command", help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            sys.exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    def _compute_args(self, parser):
        """Add common arguments for parallel commands"""
        group = parser.add_argument_group('compute parameters')
        group.add_argument('--world_size', default=1, help='Number of NCCL ranks')
        group.add_argument('--rank', default=0, help='NCCL rank')
        group.add_argument('--gpu', default="local_rank", type=str, help='GPU to use, None for CPU, "local_rank" to use value of --local_rank')
        group.add_argument('--local_rank', default=0, type=int, help='Local NCCL rank')
    def downscale(self):
        """Standardize a dataset using transforms found during atlas building"""
        import argparse, sys
        parser = argparse.ArgumentParser()
        # prefixing the argument with -- means it's optional
        parser.add_argument('input', type=str, help='Path to input image HDF5 file')
        parser.add_argument('output', type=str, help='Path to output HDF5 file')
        parser.add_argument('--h5key', default='images', help='Name of dataset in input and HDF5 files')
        parser.add_argument('--scale', default=2, type=int, help='Width of average pooling window')
        args = parser.parse_args(sys.argv[2:])

        from ..data import H5Dataset, write_dataset_h5
        dataset = H5Dataset(args.input, key=args.h5key)

        dsds = DownscaledDataset(dataset, scale=args.scale)

        write_dataset_h5(dsds, args.output, key=args.h5key)
DataCmdLine()
