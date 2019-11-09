import sys
sys.path.append('../../src/')
import autograd.numpy as np
from dOTDmodel import dOTDModel
from gendata import GenData
from rhs import rhs_CdV

if __name__ == '__main__':

    notd = 1   	   # Number of dOTD modes to be learned
    npts = 50      # Number of training points
    rhs = rhs_CdV  # Right-hand side in governing equations

    ### Generate long trajectory
    z0 = np.array([1.14,0,0,-0.91,0,0])
    ndim = z0.shape[0]
    u0 = np.random.rand(ndim,notd)
    dt = 0.05
    tf = 4000
    gen = GenData(z0, u0, tf, dt, rhs)
    t, Z, U = gen.trajectory()

    ### Generate training, validation, and testing sets
    kwargs = dict(rec=True, n_neighbors=60)
    ind_trn = np.where((t >= 1075) & (t < 1165))[0]  # Interval of blocked flow
    ind_vld = ind_trn
    a = np.floor(len(ind_trn)/(npts-1))
    ind_trn = ind_trn[::int(a)]
    ind_vld = np.setxor1d(ind_trn, ind_vld)
    ind_tst = np.where((t >= 500))[0]
    ind_tst = np.setxor1d(np.setxor1d(ind_tst, ind_trn), ind_vld)
    ind_tst = ind_tst[:-1]
    tM, zM, zdM, LM = gen.dataset(t, Z, ind_trn, **kwargs)
    tV, zV, zdV, LV = gen.dataset(t, Z, ind_vld, **kwargs)
    tS, zS, zdS, LS = gen.dataset(t, Z, ind_tst, **kwargs)

    ### Set up network, train, and predict
    layer_sizes = [ndim, 128, 128, ndim]
    step_size = 0.001
    max_iters = 5000
    lyap_off = 2000
    dOTD = dOTDModel(layer_sizes, step_size, max_iters, lyap_off)
    dOTD.train((zM, zdM, LM), notd)
    dOTD.comp_error(trn_set=(zM, zdM, LM), vld_set=(zV, zdV, LV), \
                    tst_set=(zS, zdS, LS))
    dOTD.comp_avgdist(ind_trn, ind_vld, ind_tst, Z, U)
    dOTD_tst = dOTD.predict(Z) # Predict everywhere for plotting

    ### Save for plotting
    for kk in range(notd):
        filename = 'dOTD_tst' + str(kk+1) + '.out'
        data = np.hstack((np.atleast_2d(t).T, Z, \
                          dOTD_tst[kk], U[:,:,kk]))
        np.savetxt(filename, data, fmt="%16.8e")


