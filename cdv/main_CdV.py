import sys
sys.path.append('../')
import autograd.numpy as np
import dOTD_utils as dOTD
import gendata_utils as gen
from rhs_utils import rhs_CdV
import matplotlib.pyplot as plt

if __name__ == '__main__':

    notd = 1   	   # Number of dOTD modes to solve for
    npts = 50      # Number of training points
    rhs = rhs_CdV  # Right-hand side in governing equations

## Generate long trajectory
    z0 = np.array([1.14,0,0,-0.91,0,0])
    ndim = z0.shape[0]
    u0 = np.random.rand(ndim,notd)

    dt = 0.05
    tf = 4000
    t, Z, U = gen.trajectory(z0, u0, tf, dt, rhs)

### Sample points and generate training set
    ind_trn = np.where((t >= 1075) & (t < 1165))[0]  # Pick interval of blocked flow
    a = np.floor(len(ind_trn)/(npts-1))
    ind_trn = ind_trn[::int(a)]
    tM, zM, zdM, LM = gen.dataset(t, Z, ind_trn, rhs, rec=True) # Change reconstruction here
    inputs = (zM, zdM, LM)

### Set up network and train
    layer_sizes = [ndim,128,128,ndim]
    step_size = 0.001
    max_iters = 5000
    lyap_off = 2000
    wghts_agg = dOTD.train(inputs,layer_sizes,notd,step_size,max_iters,lyap_off)

### Test on data from long trajectory
    tTest = t.reshape((len(t),1))
    xTest = Z
    dOTDtest = dOTD.test(xTest,wghts_agg)

### Save for plotting
    np.savetxt('trajectory.out', np.hstack((tTest,xTest)), delimiter='\t')
    for kk in range(notd):
        filename = 'dOTD_testing'+str(kk+1)+'.out'
        np.savetxt(filename, np.hstack((tTest,dOTDtest[kk])), delimiter='\t')
        filename = 'OTD_num'+str(kk+1)+'.out'
        np.savetxt(filename, np.hstack((tTest,U[:,:,kk])), delimiter='\t')

