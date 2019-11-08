import sys
sys.path.append('../../src/')
import autograd.numpy as np
from dOTDmodel import dOTDModel
from gendata import GenData
from rhs import rhs_3D


if __name__ == '__main__':

    notd = 3   	   # Number of dOTD modes to be learned
    npts = 10      # Number of training points
    rhs = rhs_3D   # Right-hand side in governing equations

    ### Generate long trajectory
    mu = 0.1; sqmu = np.sqrt(mu)
    z0 = np.array([sqmu*np.cos(1), sqmu*np.sin(1), mu+1e-3])
    ndim = z0.shape[0]
    u0 = np.array([[-0.84, -0.40, -0.36], \
                   [ 0.54, -0.63, -0.55], \
                   [ 0.00, -0.65,  0.75]])
    dt = 0.01
    tf = 50
    gen = GenData(z0, u0, tf, dt, rhs)
    t, Z, U = gen.trajectory()

    ### Generate training, validation, and testing sets
    kwargs = dict(rec=True, n_neighbors=7)
    ind_trn = np.where((t >= 20) & (t < 20+2*np.pi))[0] 
    a = np.floor(len(ind_trn)/(npts-1))
    ind_trn = ind_trn[::int(a)]
    ind_vld = np.where((t >= 30) & (t < 30+2*np.pi))[0]  
    ind_tst = np.where((t >= 10))[0]  
    ind_tst = np.setxor1d(np.setxor1d(ind_tst, ind_trn), ind_vld)
    ind_tst = ind_tst[:-1]
    tM, zM, zdM, LM = gen.dataset(t, Z, ind_trn, **kwargs)
    tV, zV, zdV, LV = gen.dataset(t, Z, ind_vld, **kwargs)
    tS, zS, zdS, LS = gen.dataset(t, Z, ind_tst, **kwargs)

    ### Set up network, train, and predict
    layer_sizes = [ndim, 40, ndim]
    step_size = 0.04
    max_iters = 3000
    lyap_off = 1000
    dOTD = dOTDModel(layer_sizes, step_size, max_iters, lyap_off)
    dOTD.train((zM, zdM, LM), notd)
    dOTD.comp_error(trn_set=(zM, zdM, LM), vld_set=(zV, zdV, LV), \
                    tst_set=(zS, zdS, LS))
    dOTD.comp_avgdist(ind_trn, ind_vld, ind_tst, Z, U)
    dOTD_tst = dOTD.predict(zS)
    lyap_tst = dOTD.comp_lyap(zS, LS)

    ### Save for plotting
    for kk in range(notd):
        filename = 'dOTD_tst' + str(kk+1) + '.out'
        data = np.hstack((np.atleast_2d(tS).T, zS, \
                          np.atleast_2d(lyap_tst[kk]).T, \
                          dOTD_tst[kk], U[ind_tst,:,kk]))
        np.savetxt(filename, data, fmt="%16.8e")


