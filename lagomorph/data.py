import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py
from .utils import tqdm, Tool

class H5Dataset(Dataset):
    """
    PyTorch generic Dataset for HDF5 files with dataset having first dimension
    corresponding to subject.
    """
    def __init__(self, h5path, key='images', return_indices=False, dtype=None,
            force_dim=None):
        self.h5path = h5path
        if not isinstance(key, tuple):
            key = (key,)
        self.key = key
        self.return_indices = return_indices
        if not isinstance(dtype, tuple):
            dtype = tuple([dtype for k in key])
        self.dtype = dtype
        self.force_dim = force_dim

        # check that lengths of all datasets match
        with h5py.File(self.h5path, 'r') as f:
            l = None
            for k in self.key:
                if l is None:
                    l = f[k].shape[0]
                elif f[k].shape[0] != l:
                    raise Exception(f"Mismatched lengths of datasets with keys {key}")
    def __len__(self):
        with h5py.File(self.h5path, 'r') as f:
            return f[self.key[0]].shape[0]
    def __getitem__(self, idx): 
        Is = []
        for i, (k, dt) in enumerate(zip(self.key, self.dtype)):
            with h5py.File(self.h5path, 'r') as f:
                I = torch.as_tensor(f[k][idx,...])
            if i == 0: # forcing applies only to first key
                if dt is not None:
                    I = I.type(dt)
                if self.force_dim is not None:
                    # prepend dimensions to force dimension
                    if len(I.shape) > self.force_dim+1:
                        raise Exception(f"Cannot force dimension to {self.force_dim} from {len(I.shape)}")
                    while len(I.shape) < self.force_dim+1:
                        I = I.unsqueeze(0)
            Is.append(I)
        if len(Is) == 1:
            Is = Is[0]
        if self.return_indices:
            return idx, Is
        else:
            return Is


class DownscaledDataset(Dataset):
    def __init__(self, dataset, scale, device='cuda'):
        self.dataset = dataset
        self.scale = scale
        self.device = device
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        J = self.dataset[idx]
        J = J.to(self.device)
        if J.dtype not in [torch.float32, torch.float64]:
            J = J.type(torch.float32)
        if J.dim() == 4:
            J = F.avg_pool3d(J, self.scale)
        elif J.dim() == 3:
            J = F.avg_pool2d(J, self.scale)
        return J

def batch_average(dataloader, dim=0, returns_indices=False):
    """Compute the average using streaming batches from a dataloader along a given dimension"""
    avg = None
    dtype = None
    sumsizes = 0
    comp = 0.0 # Kahan sum compensation
    with torch.no_grad():
        for img in tqdm(dataloader, 'image avg'):
            if returns_indices:
                _, img = img
            sz = img.shape[dim]
            if dtype is None:
                dtype = img.dtype
            # compute averages in float64
            avi = img.type(torch.float64).sum(dim=0)
            if avg is None:
                avg = avi/sz
            else:
                # add similar-sized numbers using this running average
                avg = avg*(sumsizes/(sumsizes+sz)) + avi/(sumsizes+sz)
            sumsizes += sz
        if dtype in [torch.float32, torch.float64]:
            # if original data in float format, return in same dtype
            avg = avg.type(dtype)
        return avg


def write_dataset_h5(dataset, h5path, key='images'):
    """
    Given a PyTorch Dataset or array_like, write in lagomorph's standard HDF5
    format.

    We assume that the dataset returns either a single image, or a tuple whose
    first entry is an image. For example, in order to return both an image and a
    set of labels, the dataset can return those as a pair of torch Tensors. Note
    that the names of the extra members of the tuple can be overridden with the
    argument 'extra_keys'.
    """
    from .utils import tqdm
    if not isinstance(key, tuple):
        # make key a tuple if it's not already
        key = (key,)
    with h5py.File(h5path, 'w') as f:
        # determine size needed for h5 dataset
        ds0 = dataset[0]
        if not isinstance(ds0, tuple):
            ds0 = (ds0,)
        # check that the length of the tuple matches args
        if len(ds0) != len(key):
            raise Exception(f"Dataset returns tuple with {len(ds0)} entries, "
                    "but only {len(key)} keys given")
        ds = []
        for d,k in zip(ds0,key):
            dtype = d.dtype
            if isinstance(d, torch.Tensor): # need a numpy dtype for h5py
                dtype = d.view(-1)[0].cpu().numpy().dtype
            sh = d.shape
            ds.append(f.create_dataset(k, shape=(len(dataset),*sh), dtype=dtype,
                    chunks=(1,*sh), compression='lzf'))
        for i in tqdm(range(len(dataset))):
            di = dataset[i]
            for I,dsi in zip(di,ds):
                if isinstance(I, torch.Tensor):
                    I = I.cpu().numpy()
                dsi[i,...] = I

class _Tool(Tool):
    """Generic dataset utilities not specific to one class of registration methods"""
    module_name = "lagomorph data"
    subcommands = ["average", "downscale"]
    def average(self):
        """Average a dataset inside an HDF5 file in the first dimension"""
        import argparse, sys
        parser = self.new_parser('average')
        # prefixing the argument with -- means it's optional
        parser.add_argument('input', type=str, help='Path to input image HDF5 file')
        parser.add_argument('output', type=str, help='Path to output HDF5 file')
        parser.add_argument('--h5key', default='images', help='Name of dataset in input HDF5 file')
        parser.add_argument('--output_h5key', default='average_image', help='Name of dataset in output HDF5 file')
        parser.add_argument('--loader_workers', default=8, type=int, help='Number of concurrent workers for dataloader')
        parser.add_argument('--batch_size', default=50, type=int, help='Batch size')
        args = parser.parse_args(sys.argv[2:])

        dataset = H5Dataset(args.input, key=args.h5key, return_indices=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.loader_workers,
                                pin_memory=True, drop_last=False)
        Iav = batch_average(dataloader, returns_indices=False)
        with h5py.File(args.output,'w') as f:
            ds = f.create_dataset(args.output_h5key, data=Iav.unsqueeze(0))
            self._stamp_dataset(ds, args)
    def downscale(self):
        """Downscale an image dataset using average pooling"""
        import argparse, sys
        parser = self.new_parser('downscale')
        # prefixing the argument with -- means it's optional
        parser.add_argument('input', type=str, help='Path to input image HDF5 file')
        parser.add_argument('output', type=str, help='Path to output HDF5 file')
        parser.add_argument('--h5key', default='images', help='Name of dataset in input and HDF5 files')
        parser.add_argument('--scale', default=2, type=int, help='Width of average pooling window')
        parser.add_argument('--copy_other_keys', action='store_true', help='Copy all other keys from input file into output verbatim')
        args = parser.parse_args(sys.argv[2:])

        dataset = H5Dataset(args.input, key=args.h5key)

        dsds = DownscaledDataset(dataset, scale=args.scale)

        write_dataset_h5(dsds, args.output, key=args.h5key)
        with h5py.File(args.output, 'a') as f:
            self._stamp_dataset(f[args.h5key], args)
        if args.copy_other_keys:
            with h5py.File(args.input, 'r') as fi, h5py.File(args.output, 'a') as fo:
                for k in tqdm(fi.keys(), desc='other keys'):
                    if k != args.h5key:
                        fi.copy(k, fo)

