import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py

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
        if J.dim() == 4:
            J = F.avg_pool3d(J, self.scale)
        elif J.dim() == 3:
            J = F.avg_pool2d(J, self.scale)
        return J


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
    from ..utils import tqdm
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

