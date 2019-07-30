import autograd.numpy as np

def rhs_3D(z):

    mu = 0.1
    
    # RHS
    y = np.zeros((3,))
    y[0] = mu*z[0] - z[1] - z[0]*z[2]
    y[1] = mu*z[1] + z[0] - z[1]*z[2]
    y[2] = -z[2] + z[0]**2 + z[1]**2

    # Linearized operator
    L = np.zeros((3,3))
    L[0,0], L[0,1], L[0,2] = mu-z[2],      -1, -z[0] 
    L[1,0], L[1,1], L[1,2] =       1, mu-z[2], -z[1]
    L[2,0], L[2,1], L[2,2] =  2*z[0],  2*z[1],    -1  

    return y, L


def rhs_CdV(z):

    zstar1 = 0.95 
    zstar4 =-0.76095 
    C      = 0.1 
    beta   = 1.25 
    gamma  = 0.2 
    b      = 0.5 
    ep     = 16*np.sqrt(2)/(5*np.pi)

    m = np.array([1,2])
    alpha = 8*np.sqrt(2)*m**2*(b**2+m**2-1)/np.pi/(4*m**2-1)/(b**2+m**2)
    beta  = beta*b**2/(b**2+m**2)
    delta = 64*np.sqrt(2)*(b**2-m**2+1)/15/np.pi/(b**2+m**2)
    gamma_m = gamma*4*np.sqrt(2)*m**3*b/np.pi/(4*m**2-1)/(b**2+m**2)
    gamma_mstar = gamma*4*np.sqrt(2)*m*b/np.pi/(4*m**2-1)

    # RHS
    y = np.zeros((6,))
    y[0] = gamma_mstar[0]*z[2] - C*(z[0]-zstar1);
    y[1] = -(alpha[0]*z[0]-beta[0])*z[2]               - C*z[1] - delta[0]*z[3]*z[5];
    y[2] =  (alpha[0]*z[0]-beta[0])*z[1] - gamma_m[0]*z[0] - C*z[2] + delta[0]*z[3]*z[4];
    y[3] = gamma_mstar[1]*z[5] - C*(z[3]-zstar4) + ep*(z[1]*z[5]-z[2]*z[4]);
    y[4] = -(alpha[1]*z[0]-beta[1])*z[5]               - C*z[4] - delta[1]*z[3]*z[2];
    y[5] =  (alpha[1]*z[0]-beta[1])*z[4] - gamma_m[1]*z[3] - C*z[5] + delta[1]*z[3]*z[1];

    # Linearized operator
    L = np.zeros((6,6))
    L[0,0], L[0,2] = -C, gamma_mstar[0]
    L[1,0], L[1,1], L[1,2], L[1,3], L[1,5] = -alpha[0]*z[2], -C, -(alpha[0]*z[0]-beta[0]), -delta[0]*z[5], -delta[0]*z[3] 
    L[2,0], L[2,1], L[2,2], L[2,3], L[2,4] = alpha[0]*z[1]-gamma_m[0], alpha[0]*z[0]-beta[0], -C, delta[0]*z[4], delta[0]*z[3]
    L[3,1], L[3,2], L[3,3], L[3,4], L[3,5] = ep*z[5], -ep*z[4], -C, -ep*z[2], gamma_mstar[1]+ep*z[1] 
    L[4,0], L[4,2], L[4,3], L[4,4], L[4,5] = -alpha[1]*z[5], -delta[1]*z[3], -delta[1]*z[2], -C, -(alpha[1]*z[0]-beta[1])
    L[5,0], L[5,1], L[5,3], L[5,4], L[5,5] = alpha[1]*z[4], delta[1]*z[3], -gamma_m[1]+delta[1]*z[1], alpha[1]*z[0]-beta[1],-C
 
    return y, L





