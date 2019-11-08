import numpy as np
from sklearn.neighbors import NearestNeighbors


class GenData:

    def __init__(self, z0, u0, tf, dt, rhs):
        self.z0 = z0
        self.u0 = u0
        self.tf = tf
        self.dt = dt
        self.rhs = rhs


    def trajectory(self, verbose=True):
        """Generate long trajectory using Adams-Bashforth time-stepping.""" 

        if verbose:
            print('Generating long trajectory...')

        z = self.z0
        u = self.u0
        ndim = z.shape[0] 
        notd = u.shape[1]
        z1=z; z2=z; z3=z 
        u1=u; u2=u; u3=u 
        nsteps = int(self.tf/self.dt)

        t = np.zeros((nsteps,))
        Z = np.zeros((nsteps,ndim))
        U = np.zeros((nsteps,ndim,notd))

        for ii in range(nsteps):

            if ii==0: c0=1.; c1=0.; c2=0.; # AB1
            if ii==1: c0=1.5; c1=-.5; c2=0; # AB2
            if ii >1: c0=23./12; c1=-16./12; c2=5./12; #AB3

            z0, L = self.rhs(z)
            phi = self.phimat(u,L)
            u0 = np.dot( np.eye(ndim) - np.dot(u,u.T), np.dot(L,u) ) \
               - np.dot(u,phi)

            fz = c0*z0 + c1*z1 + c2*z2
            fu = c0*u0 + c1*u1 + c2*u2

            z2 = z1; z1 = z0 
            u2 = u1; u1 = u0 

            z = z + self.dt*fz
            u = u + self.dt*fu 
            u, R = self.gram_schmidt(u)

            t[ii] = ii*self.dt
            Z[ii,:] = z
            U[ii,:,:] = u

        return t, Z, U


    def gram_schmidt(self, A):
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


    def phimat(self, u, L):
        """Compute the \Phi tensor in OTD equations."""
        r = u.shape[1]
        phi = np.zeros((r,r))
        for ii in range(r):
            for kk in range(r):
                if kk<ii: phi[ii,kk] = -(u[:,kk].T).dot(L.T).dot(u[:,ii])
                if kk>ii: phi[ii,kk] =  (u[:,kk].T).dot(L  ).dot(u[:,ii])
        return phi


    def dataset(self, t, Z, ind_trn, rec=False, n_neighbors=60, verbose=True):
        """Form training set by sampling long trajectory and 
        reconstruct F(x) and L(x).""" 

        if verbose:
            print('Generating dataset...')

        ndim = Z.shape[1]
        FZ = np.zeros((len(ind_trn), ndim))
        LZ = np.zeros((len(ind_trn), ndim, ndim))

        if rec:
        # Reconstruct F(x) and L(x) using K-NN
            if verbose:
                print('Computing K-NN...')
            FZ = (Z[ind_trn+1,:]-Z[ind_trn,:])/self.dt
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, \
                                    algorithm='ball_tree').fit(Z)
            distances, indices = nbrs.kneighbors()
            indices[np.where(indices==Z.shape[0]-1)] = Z.shape[0]-2
            for ii, ind in enumerate(ind_trn):
                v0 = Z[indices[ind,:]  ,:] - Z[ind  ,:]; v0=v0.T
                v1 = Z[indices[ind,:]+1,:] - Z[ind+1,:]; v1=v1.T
                dv = (v1-v0)/self.dt
                Lz = np.dot(dv,np.linalg.pinv(v0))
                LZ[ii,:,:] = Lz

        else:
        # Otherwise, use equations
            for ii, ind in enumerate(ind_trn):
                Fz, Lz = self.rhs(Z[ind,:])
                FZ[ii,:] = Fz
                LZ[ii,:,:] = Lz

        return t[ind_trn], Z[ind_trn], FZ, LZ


