import sys
sys.path.append('../')
import autograd.numpy as np
import dOTD_utils as dOTD
import gendata_utils as gen
from rhs_utils import rhs_3D
import matplotlib.pyplot as plt


if __name__ == '__main__':

    notd = 3   	   # Number of dOTD modes to solve for
    npts = 10      # Number of training points
    rhs = rhs_3D   # Right-hand side in governing equations

### Generate long trajectory
    mu = 0.1; sqmu = np.sqrt(mu)
    z0 = np.array([sqmu*np.cos(1),sqmu*np.sin(1),mu])
    ndim = z0.shape[0]
    u0 = np.random.rand(ndim,notd)
    dt = 0.01
    tf = 600
    t, Z, U = gen.trajectory(z0, u0, tf, dt, rhs)

### Sample points and generate training set
    ind_trn = np.where((t >= 500) & (t < 500+2*np.pi))[0]  # Pick one period
    a = np.floor(len(ind_trn)/(npts-1))
    ind_trn = ind_trn[::int(a)]
    tM, zM, zdM, LM = gen.dataset(t, Z, ind_trn, rhs)
    inputs = (zM, zdM, LM)

### Set up network and train
    layer_sizes = [ndim,20,ndim]
    step_size = 0.04
    max_iters = 2000
    wghts_agg = dOTD.train(inputs,layer_sizes,notd,step_size,max_iters)

### Test on data from long trajectory
    tTest = t.reshape((len(t),1))
    xTest = Z
    dOTDtest = dOTD.test(xTest,wghts_agg)

### Save for plotting
    np.savetxt('training_stmps.txt', ind_trn)  # Training stamps
    for kk in range(notd):
        filename = 'dOTD_testing'+str(kk+1)+'.out'
        np.savetxt(filename, np.hstack((tTest,dOTDtest[kk])), delimiter='\t')
        filename = 'OTD_num'+str(kk+1)+'.out'
        np.savetxt(filename, np.hstack((tTest,U[:,:,kk])), delimiter='\t')

