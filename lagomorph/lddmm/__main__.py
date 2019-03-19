from .__init__ import *

from lagomorph_cuda import set_debug_mode

set_debug_mode(True)

class LDDMMCmdLine():
    # Customized from:
    # https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html#
    def __init__(self):
        import argparse, sys
        subcmds = ['atlas']
        subhelps = '\n'.join([f'  {c}: {getattr(self,c).__doc__}' for c in subcmds])
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                usage=f"python -m lagomorph.lddmm <command> [<args>]\n\nAvailable subcommands:\n{subhelps}")
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
    def _stamp_dataset(self, ds, args):
        from ..version import __version__
        import json
        ds.attrs['lagomorph_version'] = __version__
        ds.attrs['command_args'] = json.dumps(vars(args))
    def atlas(self):
        """
        Build LDDMM atlas from HDF5 image dataset.

        This command will result in a new HDF5 file containing the following datasets:
            atlas: the atlas image
            momenta: a momentum vector field for each input image
            epoch_losses: mean squared error + regularization terms averaged across epochs (this is just an average of the iteration losses per epoch)
            iter_losses: loss at each iteration

        Note that metadata like the lagomorph version and parameters this
        command was invoked with are attached to the 'atlas' dataset as
        attributes.
        """
        import argparse, sys
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # prefixing the argument with -- means it's optional
        dg = parser.add_argument_group('data parameters')
        dg.add_argument('input', type=str, help='Path to input image HDF5 file')
        dg.add_argument('--force_dim', default=None, type=int, help='Force dimension of images instead of determining based on dataset shape')
        dg.add_argument('--h5key', '-k', default='images', help='Name of dataset in input HDF5 file')
        dg.add_argument('--loader_workers', default=8, help='Number of concurrent workers for dataloader')
        dg.add_argument('output', type=str, help='Path to output HDF5 file')

        ag = parser.add_argument_group('algorithm parameters')
        ag.add_argument('--num_epochs', default=1000, type=int, help='Number of epochs')
        ag.add_argument('--batch_size', default=50, type=int, help='Batch size')
        #ag.add_argument('--image_update_freq', default=0, type=int, help='Update base image every N iterations. 0 for once per epoch')
        ag.add_argument('--lddmm_steps', default=1, type=int, help='LDDMM gradient steps to take each iteration')
        ag.add_argument('--deformation_scale', default=1, type=int, help='Amount to downscale grid for LDDMM momenta/deformation')
        ag.add_argument('--reg_weight', default=1e-1, type=float, help='Amount of regularization for deformations')
        ag.add_argument('--learning_rate_m', default=1e-3, type=float, help='Learning rate for momenta')
        ag.add_argument('--learning_rate_I', default=1e5, type=float, help='Learning rate for atlas image')

        self._compute_args(parser)
        args = parser.parse_args(sys.argv[2:])

        if args.gpu == 'local_rank':
            args.gpu = args.local_rank
        else:
            args.gpu = int(args.gpu)

        torch.cuda.set_device(args.local_rank)

        if args.world_size > 1:
            torch.distributed.init_process_group(backend='nccl',
                    world_size=args.world_size, init_method='env://')

        from ..data import H5Dataset
        dataset = H5Dataset(args.input, key=args.h5key, return_indices=True,
                force_dim=args.force_dim)

        if args.deformation_scale != 1:
            im0 = dataset[0][1]
            momentum_shape = [s//args.deformation_scale for s in im0.shape[1:]]
            del im0
        else:
            momentum_shape = None

        I, momenta, epoch_losses, iter_losses = lddmm_atlas(dataset,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lddmm_steps=args.lddmm_steps,
                #image_update_freq=args.image_update_freq,
                momentum_shape=momentum_shape,
                reg_weight=args.reg_weight,
                learning_rate_pose=args.learning_rate_m,
                learning_rate_image=args.learning_rate_I,
                loader_workers=args.loader_workers,
                world_size=args.world_size,
                rank=args.rank,
                gpu=args.gpu)

        import h5py
        with h5py.File(args.output, 'w') as f:
            atds = f.create_dataset('atlas', data=I.cpu().numpy())
            self._stamp_dataset(atds, args)
            f.create_dataset('momenta', data=momenta.cpu().numpy())
            f.create_dataset('epoch_losses', data=np.asarray(epoch_losses))
            f.create_dataset('iter_losses', data=np.asarray(iter_losses))
LDDMMCmdLine()
