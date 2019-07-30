import numpy as np
from rhs_utils import rhs_CdV, rhs_3D
from sklearn.neighbors import NearestNeighbors
from pprint import pprint

def gram_schmidt(A):
    """Gram-Schmidt orthonormalization based on Gil Strang's book."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = Q[:, i].dot(v)
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R


def phimat(u, L):
    """Compute the \Phi tensor in OTD equations."""
    r = u.shape[1]
    phi = np.zeros((r,r))
    for ii in range(r):
        for kk in range(r):
            if kk<ii: phi[ii,kk] = -(u[:,kk].T).dot(L.T).dot(u[:,ii])
            if kk>ii: phi[ii,kk] =  (u[:,kk].T).dot(L  ).dot(u[:,ii])
    return phi


def trajectory(z, u, tf, dt, get_rhs):
    """Generate long trajectory using Adams-Bashforth time-stepping.""" 
    print('Generating long trajectory...')
    ndim = z.shape[0] 
    notd = u.shape[1]
    z1=z; z2=z; z3=z 
    u1=u; u2=u; u3=u 
    nsteps = int(tf/dt)

    t = np.zeros((nsteps,))
    Z = np.zeros((nsteps,ndim))
    U = np.zeros((nsteps,ndim,notd))

    for ii in range(nsteps):

        if ii==0: c0=1.; c1=0.; c2=0.; # AB1
        if ii==1: c0=1.5; c1=-.5; c2=0; # AB2
        if ii >1: c0=23./12; c1=-16./12; c2=5./12; #AB3

        z0, L = get_rhs(z)
        phi = phimat(u,L)
        u0 = np.dot( np.eye(ndim) - np.dot(u,u.T), np.dot(L,u) ) \
           - np.dot(u,phi)

        fz = c0*z0 + c1*z1 + c2*z2
        fu = c0*u0 + c1*u1 + c2*u2

        z2 = z1; z1 = z0 
        u2 = u1; u1 = u0 

        z = z + dt*fz
        u = u + dt*fu 
        u,R = gram_schmidt(u)

        t[ii] = ii*dt
        Z[ii,:] = z
        U[ii,:,:] = u

    return t, Z, U


def dataset(t, Z, ind_trn, get_rhs, rec=False, n_neighbors=60):
    """Form training set by sampling long trajectory and reconstruct F(x) and L(x).""" 
    print('Generating dataset...')
    ndim = Z.shape[1]
    FZ = np.zeros((len(ind_trn), ndim))
    LZ = np.zeros((len(ind_trn), ndim, ndim))

    if rec:
    # Reconstruct F(x) and L(x) using K-NN
        print('Computing K-NN...')
        dt = t[1]-t[0]
        FZ = (Z[ind_trn+1,:]-Z[ind_trn,:])/dt
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(Z)
        distances, indices = nbrs.kneighbors()
        for ii, ind in enumerate(ind_trn):
            v0 = Z[indices[ind,:]  ,:] - Z[ind  ,:]; v0=v0.T
            v1 = Z[indices[ind,:]+1,:] - Z[ind+1,:]; v1=v1.T
            dv = (v1-v0)/dt
            Lz = np.dot(dv,np.linalg.pinv(v0))
            LZ[ii,:,:] = Lz

    else:
    # Otherwise, use equations
        for ii, ind in enumerate(ind_trn):
            Fz, Lz = get_rhs(Z[ind,:])
            FZ[ii,:] = Fz
            LZ[ii,:,:] = Lz

    return t[ind_trn], Z[ind_trn], FZ, LZ


