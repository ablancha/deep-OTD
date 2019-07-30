import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, jacobian
import autograd.misc.optimizers 
from autograd.misc import flatten
from autograd.wrap_util import wraps
import _pickle as cPickle


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
        g = grad(x,i)
        if callback:
            loss=callback(x, i, g)
            if loss<tol: break # Break if error less than tol
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
    # Add learning rate schedule here, e.g.:
        #  if i==4000: step_size=step_size*0.5
        #  if i==5000: step_size=step_size*0.5
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return x


def init_w(layer_sizes, rs=npr.RandomState( )):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * np.sqrt(2/(insize + outsize)),   
             rs.randn(outsize) * 0.0)
             for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]


def nnet(wghts, inputs, wghts_agg, nonlinearity=np.tanh):
    """Predict output with NN."""
    X = inputs
    for W, b in wghts:
        outputs = np.dot(inputs, W) + b
        inputs = nonlinearity(outputs)
    Y = gs(outputs,X,wghts_agg)
    return Y


def gs(outputs,X,wghts_agg):
    """Orthonormalize k-th output against first k-1 outputs."""
    outdim=outputs.ndim
    if outdim==1: outputs=np.vstack((outputs,outputs))
    kotd=len(wghts_agg)
    for ii in range(kotd):
      a = nnet(wghts_agg[ii], X, wghts_agg[0:ii])
      if a.ndim==1: a=np.vstack((a,a))
      rij=np.sum(a*outputs, axis=1)
      outputs = outputs - rij[:, np.newaxis]*a
    asq=np.sum(outputs*outputs,axis=1)
    outputs=outputs/np.sqrt(asq[:, np.newaxis])
    if outdim==1: outputs=outputs[0,:]
    return outputs


def loss(wghts,step,gargs):
    """Compute OTD loss function."""
    xM, xdM, LM = gargs[0]
    wghts_agg = gargs[1]
    lyap_off = gargs[2]
    gargs_batch=gargs

    ifbatch = False     # Mini-batching
    if (ifbatch):
      idx = batch_indices(step)
      gargs_batch = [ (xM[idx], xdM[idx], LM[idx]), wghts_agg ]

    l_pde, l_lya = losses(wghts,step,gargs_batch)
    if step >= lyap_off: l_lya=0.0  # On/Off switch

    return l_pde + l_lya


def losses(wghts,step,gargs_batch):
    """Compute PDE and Lyapunov losses."""
    xM, xdM, LM = gargs_batch[0]      # (Un)batched data
    wghts_agg = gargs_batch[1]        # Previous modes
    kotd = len(wghts_agg)
    npts, ndim = np.shape(xM)
    nn_out_jacobian = jacobian(nnet,1)

    l_pde=0.0

    OUT  = nnet(wghts, xM, wghts_agg)
    LU   = np.einsum('abi,ai->ab',  LM, OUT)
    UtLU = np.einsum('ai,ai->a'  , OUT,  LU)
    UUtLU= OUT*UtLU[:,np.newaxis]
    RHS  = LU-UUtLU
    for jj in range(kotd):
        OUTjj = nnet(wghts_agg[jj], xM, wghts_agg[0:jj])
        LUjj  = np.einsum('abi,ai->ab',    LM, OUTjj)
        PROJ1 = np.einsum('ai,ai->a'  , OUTjj,    LU)
        PROJ2 = np.einsum('ai,ai->a'  ,   OUT,  LUjj)
        RHS   = RHS - OUTjj*(PROJ1[:,np.newaxis]+PROJ2[:,np.newaxis])

    for ii in range(npts):
        dudx = jacobian(nnet,1)(wghts, xM[ii,:], wghts_agg)
        LHS  = np.dot(dudx,xdM[ii,:])
        l_pde= l_pde + np.sum((LHS-RHS[ii,:])**2)/npts

    l_lya = -np.sinh(np.mean(UtLU))     # Lyapunov regularization

    return l_pde, l_lya


def callback(wghts, step, g, gargs):
    """Callback function for optimization loop."""
    closs=1.0
    if step == 0:
        print('/-------------------------/')
        print('/ Solving for dOTD mode {0:d} /'.format(len(gargs[1])+1))
        print('/-------------------------/')
    if step % 50 == 0:
        closs, lyap = losses(wghts,step,gargs)    # PDE loss for current mode
        logstring = "Iteration {0:4d} \t  Loss PDE = {1:0.12f} \t Lyap = {2:0.2e} "\
                    .format(step,closs,-np.arcsinh(lyap))
        print(logstring)
        with open('logerr.out', "a") as text_file:  # Write to logfile
            text_file.write(logstring)
    return closs


def train(inputs, layer_sizes, notd, step_size, num_iters, lyap_off, saveWeights=True):
    """Run optimization loop."""
    wghts_agg = []
    for kk in range(notd):
        if kk+1 == layer_sizes[0]: num_iters=1  # Do only one iteration if notd=ndim
        wghts = init_w(layer_sizes)
        gargs = [inputs, wghts_agg, lyap_off]
        wghts = myadam(grad(loss), wghts, gargs,
                       callback=callback, step_size=step_size, num_iters=num_iters)
        wghts_agg.append(wghts)
    if saveWeights:
        cPickle.dump(wghts_agg, open('wghts_trained.pck', 'wb'))  # Save weights
    return wghts_agg


def test(xTest, wghts_agg):
    """Test on unseen data."""
    print('Testing model...')
    notd=len(wghts_agg)
    otd_agg=[]
    for kk in range(notd):
        nn=nnet(wghts_agg[kk], xTest, wghts_agg[0:kk])
        otd_agg.append(nn)
    return otd_agg


