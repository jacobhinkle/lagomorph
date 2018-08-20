from pycuda import gpuarray
from matplotlib import pyplot as plt
import numpy as np

from .deform import identity

def gridplot(u, Nx=64, Ny=64, displacement=True, color='black', **kwargs):
    """Given a displacement field, plot a displaced grid"""
    assert u.shape[0] == 1, "Only send one deformation at a time"
    if isinstance(u, gpuarray.GPUArray):
        u = u.get()
    if Nx is None:
        Nx = u.shape[2]
    if Ny is None:
        Ny = u.shape[3]
    # downsample displacements
    h = np.copy(u[:,:,::u.shape[2]//Nx, ::u.shape[3]//Ny])
    # adjust displacements for downsampling
    h[:,0,...] /= float(u.shape[2]//Nx)
    h[:,1,...] /= float(u.shape[3]//Ny)
    if displacement: # add identity
        h[0,0,...] += np.arange(Nx).reshape((Nx,1))
        h[0,1,...] += np.arange(Ny).reshape((1,Ny))
    # put back into original index space
    h[:,0,...] *= float(u.shape[2]//Nx)
    h[:,1,...] *= float(u.shape[3]//Ny)
    # create a meshgrid of locations
    for i in range(h.shape[2]):
        plt.plot(h[0,1,i,:], h[0,0,i,:], color=color, **kwargs)
    for i in range(h.shape[3]):
        plt.plot(h[0,1,:,i], h[0,0,:,i], color=color, **kwargs)
    plt.axis('equal')
    plt.gca().invert_yaxis()

def quiver(u, Nx=32, Ny=32, color='black', units='xy', angles='xy', scale=1.0, **kwargs):
    """Given a displacement field, plot a quiver of vectors"""
    assert u.shape[0] == 1, "Only send one deformation at a time"
    if isinstance(u, gpuarray.GPUArray):
        u = u.get()
    if Nx is None:
        Nx = u.shape[2]
    if Ny is None:
        Ny = u.shape[3]
    # downsample displacements
    h = np.copy(u[:,:,::u.shape[2]//Nx, ::u.shape[3]//Ny])
    ix = identity(u.shape, u.dtype)[:,:,::u.shape[2]//Nx, ::u.shape[3]//Ny]
    # create a meshgrid of locations
    plt.quiver(ix[0,1,:,:], ix[0,0,:,:], h[0,1,:,:], h[0,0,:,:], color=color,
               angles=angles, units=units, scale=scale, **kwargs)
    plt.axis('equal')
    plt.gca().invert_yaxis()
