# regrid_risr_interp_for_gemini.py

import numpy as np
import h5py
from scipy.interpolate import interpn

grid_filename = 'gridxyz.h5'
interp_filename = 'sample_gemini_event.h5'
output_filename = 'nexyz.h5'

# define modified chapman function
def chapman_piecewise(z, A, Ht, Hb, z0):

    # From Schunk and Nagy, 2009; eqn 11.57
    # Topside
    zp = (z-z0)/Ht
    NeT = A*np.exp(0.5*(1-zp-np.exp(-zp)))

    # Bottomside
    zp = (z-z0)/Hb
    NeB = A*np.exp(1-zp-np.exp(-zp))
    # NOTE: The topside and bottomside functions have different shaping parameters (0.5 and 1).
    #   This is to replicate the fitting funciton in the original interpolation algorithm which
    #   contains this bug.

    Ne = NeB.copy()
    Ne[z.flatten()>z0,:,:] = NeT[z.flatten()>z0,:,:]

    return Ne

# Read in original interpolated density values
with h5py.File(interp_filename, 'r') as h5:
    xgrid0 = h5['XGrid'][:]*1000.
    ygrid0 = h5['YGrid'][:]*1000.
    NmF2 = h5['NmF2'][:]
    hmF2 = h5['hmF2'][()]
    HmT = h5['HmT'][()]
    HmB = h5['HmB'][()]

# Read in new grid
with h5py.File(grid_filename,"r") as f:
    newz=f["z"][:]
    newx=f["x"][:]
    newy=f["y"][:]

newxgrid, newygrid = np.meshgrid(newx, newy)

# interpolate to new grid
interp_NmF2 = interpn((xgrid0[0,:], ygrid0[:,0]), NmF2, (newxgrid.flatten(), newygrid.flatten()), method='linear', bounds_error=False, fill_value=2.e11)
interp_NmF2 = interp_NmF2.reshape(newxgrid.shape)

interp_dens = chapman_piecewise(newz[:,None,None], interp_NmF2[None,:], HmT, HmB, hmF2)

# # Validate with plots
# import matplotlib.pyplot as plt
# for dslice in interp_dens:
#     c = plt.pcolormesh(newxgrid, newygrid, dslice, vmin=0, vmax=5.e11)
#     plt.colorbar(c)
#     plt.show()

with h5py.File(output_filename, 'w') as h5:
    h5.create_dataset('Ne', data=interp_dens)
