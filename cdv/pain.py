import sys
sys.path.append('../')
import autograd.numpy as np
import dOTD_utils as dOTD
import gendata_utils as gen
from rhs_utils import rhs_CdV
import matplotlib.pyplot as plt

def load_data(filename,npts):
    """Load CdV data."""
    ndim=6
    data=np.loadtxt(filename)
    data=data[10000:11200]
    ind=np.linspace(0,len(data),npts,endpoint=False,dtype=int)
    data=data[ind,:]    # Uniform in time
    tM =data[:,                0]; print(tM)
    xM =data[:,       1:  ndim+1]
    xdM=data[:,  ndim+1:2*ndim+1];
    LM = np.reshape(data[:,2*ndim+1:       :], (len(data),ndim,ndim), order='F')
    return xM, xdM, LM, tM.reshape((len(data),1))


def load_testdata(filename):
    """Load CdV data."""
    ndim=6
    data=np.loadtxt(filename)
    data=data[8000:40000]
    tM =data[:,                0]
    xM =data[:,       1:  ndim+1]
    xdM=data[:,  ndim+1:2*ndim+1]
    LM =data[:,2*ndim+1:       :]
    return xM, xdM, LM, tM.reshape((len(data),1))


if __name__ == '__main__':

    notd = 1   	   # Number of dOTD modes to solve for
    npts = 50      # Number of training points
    rhs = rhs_CdV  # Right-hand side in governing equations

    zM, zdM, LM, tM = load_data('myFileRec.txt',npts)
    ndim = zM.shape[1]
    inputs = (zM, zdM, LM)

### Set up network and train
    layer_sizes = [ndim,128,128,ndim]
    step_size = 0.001
    max_iters = 5000
    wghts_agg = dOTD.train(inputs,layer_sizes,notd,step_size,max_iters)

### Test on data from long trajectory
    xM, xdM, LM, tM = load_testdata('myFile.txt')
    tTest = tM.reshape((len(tM),1))
    xTest = xM
    dOTDtest = dOTD.test(xTest,wghts_agg)

### Save for plotting
   #np.savetxt('training_stmps.txt', ind_trn)  # Training stamps
    for kk in range(notd):
        filename = 'dOTD_testing'+str(kk+1)+'.out'
        np.savetxt(filename, np.hstack((tTest,dOTDtest[kk])), delimiter='\t')
   #    filename = 'OTD_num'+str(kk+1)+'.out'
   #    np.savetxt(filename, np.hstack((tTest,U[:,:,kk])), delimiter='\t')

