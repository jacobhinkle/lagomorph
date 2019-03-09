import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py

class H5Dataset(Dataset):
    """
    PyTorch generic Dataset for HDF5 files with dataset having first dimension
    corresponding to subject.
    """
    def __init__(self, h5path, key='images', return_indices=False, dtype=None):
        self.handle = h5py.File(h5path, 'r')
        self.ds = self.handle[key]
        self.return_indices = return_indices
        self.dtype = dtype
    def __len__(self):
        return self.ds.shape[0]
    def __getitem__(self, idx): 
        I = torch.as_tensor(self.ds[idx,...])
        if self.dtype is not None:
            I = I.type(self.dtype)
        if self.return_indices:
            return idx, I
        else:
            return I


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
    """Given a PyTorch Dataset or array_like, write in lagomorph's standard HDF5
    format"""
    from ..utils import tqdm
    with h5py.File(h5path, 'w') as f:
        # determine size needed for h5 dataset
        ds0 = dataset[0]
        dtype = ds0.dtype
        if isinstance(ds0, torch.Tensor): # need a numpy dtype for h5py
            dtype = ds0.view(-1)[0].cpu().numpy().dtype
        sh = ds0.shape
        ds = f.create_dataset(key, shape=(len(dataset),*sh), dtype=dtype,
                chunks=(1,*sh), compression='lzf')
        for i in tqdm(range(len(dataset))):
            I = dataset[i]
            if isinstance(I, torch.Tensor):
                I = I.cpu().numpy()
            ds[i,...] = I

