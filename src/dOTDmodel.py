import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, jacobian
import autograd.misc.optimizers 
from autograd.misc import flatten
from autograd.wrap_util import wraps
import os


class dOTDModel:

    def __init__(self, layer_sizes, step_size, max_iters, lyap_off):
        self.layer_sizes = layer_sizes
        self.step_size = step_size
        self.max_iters = max_iters
        self.lyap_off = lyap_off


    def init_w(self, rseed=None):
        """Build a list of (weights, biases) tuples, one for each layer,
        using Xavier initialization.
        """
        rs = npr.RandomState(rseed)
        return [ (rs.randn(insize, outsize)*np.sqrt(2/(insize+outsize)),   
                  rs.randn(outsize)*0.0)
                 for insize, outsize in zip(self.layer_sizes[:-1], 
                                            self.layer_sizes[1:]) ]


    def nnet(self, wghts, inputs, wghts_agg, nonlinearity=np.tanh):
        """Predict output with neural network and apply Gram-Schmidt."""
        X = inputs
        for W, b in wghts:
            outputs = np.dot(inputs, W) + b
            inputs = nonlinearity(outputs)
        Y = self.gs(outputs, X, wghts_agg)
        return Y


    def gs(self, outputs, X, wghts_agg):
        """Orthonormalize k-th output against first k-1 outputs."""
        outputs = np.atleast_2d(outputs)
        kotd = len(wghts_agg)
        for ii in range(kotd):
            a = self.nnet(wghts_agg[ii], X, wghts_agg[0:ii])
            rij = np.sum(a*outputs, axis=1)
            outputs = outputs - rij[:, np.newaxis]*a
        asq = np.sum(outputs*outputs, axis=1)
        outputs = outputs/np.sqrt(asq[:, np.newaxis])
        return outputs


    def loss(self, wghts, step, gargs):
        """Compute OTD loss function."""
        xM, xdM, LM = gargs[0]
        wghts_agg = gargs[1]
        gargs_batch = gargs
        ifbatch = False     # Mini-batching -- Implemented but not tested
        if (ifbatch):
            batch_size=10
            num_batches = int(np.ceil(len(xM) / batch_size))
            idd = step % num_batches
            idx = slice(idd * batch_size, (idd+1) * batch_size)
            gargs_batch = [ (xM[idx], xdM[idx], LM[idx]), wghts_agg ]
        l_pde, l_lya = self.losses(wghts, step, gargs_batch)
        if step >= self.lyap_off: 
            l_lya=0.0  # On/Off switch           
        return l_pde + l_lya


    def losses(self, wghts, step, gargs_batch):
        """Compute PDE and Lyapunov losses separately."""
        xM, xdM, LM = gargs_batch[0]      # (Un)batched data
        wghts_agg = gargs_batch[1]        # Previous modes
        kotd = len(wghts_agg)
        npts, ndim = np.shape(xM)

        l_pde=0.0
        OUT = self.nnet(wghts, xM, wghts_agg)
        LU  = np.einsum('abi,ai->ab', LM, OUT)
        UtLU = np.einsum('ai,ai->a', OUT, LU)
        UUtLU = OUT*UtLU[:,np.newaxis]
        RHS = LU - UUtLU
        for jj in range(kotd):
            OUTjj = self.nnet(wghts_agg[jj], xM, wghts_agg[0:jj])
            LUjj = np.einsum('abi,ai->ab', LM, OUTjj)
            PROJ1 = np.einsum('ai,ai->a', OUTjj, LU)
            PROJ2 = np.einsum('ai,ai->a', OUT, LUjj)
            RHS = RHS - OUTjj*(PROJ1[:,np.newaxis] + PROJ2[:,np.newaxis])
        for ii in range(npts):
            dudx = jacobian(self.nnet,1)(wghts, xM[ii,:], wghts_agg)
            LHS = np.dot(dudx,xdM[ii,:])
            l_pde = l_pde + np.sum((LHS-RHS[ii,:])**2)/npts
        l_lya = -np.sinh(np.mean(UtLU))  # Lyapunov regularization
        return l_pde, l_lya


    def callback(self, wghts, step, g, gargs):
        """Callback function for optimization loop."""
        closs=1.0
        if step == 0:
            print(' ------------------------- ')
            print('| Solving for dOTD mode {0:d} |'.format(len(gargs[1])+1))
            print(' ------------------------- ')
            if len(gargs[1]) == 0:
                try:
                    os.remove('logerr.out')
                except OSError:
                    pass
        if step % 50 == 0:
            closs, lyap = self.losses(wghts, step, gargs) 
            logstring = ("Iteration {0:4d} \t"
                       + "Loss PDE = {1:0.12f} \t"
                       + "Lyap = {2:0.2e}")\
                       .format(step, closs, -np.arcsinh(lyap))
            print(logstring)
            with open('logerr.out', "a") as text_file:  # Write to logfile
                text_file.write(logstring + "\n")
        return closs


    def train(self, inputs, notd, rseed=None, verbose=True):
        """Run optimization loop."""
        wghts_agg = []
        num_iters = self.max_iters
        callback = None
        if verbose:
            callback = self.callback

        for kk in range(notd):
            if kk+1 == self.layer_sizes[0]: 
                num_iters = 1  # Do only one iteration if notd=ndim
            wghts = self.init_w(rseed)
            gargs = [inputs, wghts_agg]
            wghts = myadam(grad(self.loss), wghts, gargs, \
                           callback=callback, \
                           step_size=self.step_size, num_iters=num_iters)
            wghts_agg.append(wghts)
        self.wghts_agg = wghts_agg

        return wghts_agg


    def predict(self, xTest):
        """Compute dOTD modes given data."""
        notd = len(self.wghts_agg)
        otd_agg = []
        for kk in range(notd):
            nn = self.nnet(self.wghts_agg[kk], xTest, self.wghts_agg[0:kk])
            otd_agg.append(nn)
        return otd_agg


    def comp_lyap(self, xTest, LTest):
        """Compute local Lyapunov exponents given data."""
        lya_agg = []
        otd_agg = self.predict(xTest)
        notd = len(otd_agg)
        for kk in range(notd):
            nn = otd_agg[kk]
            ll = np.einsum('ai,ai->a', nn, np.einsum('abi,ai->ab', LTest, nn))
            lya_agg.append(ll)
        return lya_agg


    def comp_error(self, trn_set, vld_set, tst_set, verbose=True):
        """Compute PDE loss given data."""
        notd = len(self.wghts_agg)
        dsts = [trn_set, vld_set, tst_set]
        stgs = ["Training set  ", "Validation set", "Test set      "]
        errs = [np.zeros(notd), np.zeros(notd), np.zeros(notd)]
        for kk in range(notd):
            if verbose:
                print('Error for dOTD mode {0:d}'.format(kk+1))
            for ii, (dst, stg) in enumerate(zip(dsts, stgs)):
                gargs = [dst, self.wghts_agg[0:kk]]
                l_pde, _ = self.losses(self.wghts_agg[kk], 0, gargs)
                errs[ii][kk] = l_pde
                if verbose:
                    print( ("    " + stg + "\t {0:0.2e}").format(l_pde) )
        return errs


    def comp_avgdist(self, ind_trn, ind_vld, ind_tst, Z, U, verbose=True):
        """Compute average distance given data."""
        notd = len(self.wghts_agg)
        inds = [ind_trn, ind_vld, ind_tst]
        stgs = ["Training set  ", "Validation set", "Test set      "]
        diss = [np.zeros(notd), np.zeros(notd), np.zeros(notd)]
        for kk in range(notd):
            if verbose:
                print('Average distance for dOTD mode {0:d}'.format(kk+1))
            for ii, (ind, stg) in enumerate(zip(inds, stgs)):
                uDeep = self.predict(Z[ind,:])
                dist = 1-np.abs(np.einsum('ij,ij->i', uDeep[kk], U[ind,:,kk]))
                avgdist = np.mean(dist)
                diss[ii][kk] = avgdist
                if verbose:
                    print( ("    " + stg + "\t {0:0.2e}").format(avgdist) )
        return diss




########################################################
### Helper routines from Autograd misc.optimizers.py ###
########################################################

def myunflatten_optimizer(optimize):
    """Adapted from Autograd's 'unflatten_optimizer' to account for extra arguments."""
    @wraps(optimize)
    def _optimize(grad, x0, gargs, callback=None, *args, **kwargs):
        _x0, unflatten = flatten(x0)
        _grad = lambda x, i: flatten(grad(unflatten(x), i, gargs))[0]
        if callback:
            _callback = lambda x, i, g: callback(unflatten(x), i, unflatten(g), gargs)
        else:
            _callback = None
        return unflatten(optimize(_grad, _x0, gargs, _callback, *args, **kwargs))
    return _optimize


@myunflatten_optimizer
def myadam(grad, x, gargs, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8, tol=1e-6):
    """Adapted from Autograd's Adam optimization routine."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        # Add learning rate schedule here, e.g.:
        #    if i==4000: step_size=step_size*0.5
        #    if i==5000: step_size=step_size*0.5
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
        if callback:
            loss = callback(x, i, g)
            if loss < tol: break # Break if error less than tol
    return x


