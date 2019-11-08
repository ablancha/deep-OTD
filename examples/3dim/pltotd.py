import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def latexify(ticklabels):
    """Manually set LaTeX format for tick labels."""
    return [r"$" + str(label) + "$" for label in ticklabels]
         

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 9

notd = 3
ndim = 3

time = []
traj = []
lyap = []
uDeep = []
uNum = []

for ii in range(notd):
    data = np.genfromtxt('dOTD_tst' + str(ii+1) + '.out')
    time.append(data[:,0])
    traj.append(data[:,1:ndim+1])
    lyap.append(data[:,ndim+1:ndim+2])
    uDeep.append(data[:,ndim+2:2*ndim+2])
    uNum.append(data[:,2*ndim+2::])

ind_st = 2000
xticks = [time[0][ind_st]+2*ii*np.pi for ii in range(3)]
yticks = [-1.2, 0, 1.2]

for ii in range(notd):
    fig = plt.figure(figsize=(2.2,3.8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0.1)
    for jj in range(ndim):
        ax = plt.subplot(ndim,1,jj+1)
        sc = np.sign(np.einsum('ij,ij->i', uDeep[ii], uNum[ii]))
        plt.plot(time[ii], uDeep[ii][:,jj]*sc, \
                 '-' , color="#ca0020", linewidth=0.75)
        plt.plot(time[ii], uNum[ii][:,jj], \
                 '-.', color="#0571b0", linewidth=0.75)
        plt.xlabel(r'$t-{0:0.0f}$'.format(time[0][ind_st]))
        if jj==0: plt.ylabel('$u_{' + str(ii+1) + ',x}$')
        if jj==1: plt.ylabel('$u_{' + str(ii+1) + ',y}$')
        if jj==2: plt.ylabel('$u_{' + str(ii+1) + ',z}$')
        plt.xlim(xticks[0], xticks[-1])
        plt.ylim(yticks[0], yticks[-1])
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([r'$0$', r'$2\pi$', r'$4\pi$'])
        ax.set_yticklabels(latexify(yticks))
        ax.yaxis.set_label_coords(-0.2, 0.5)
        ax.tick_params(direction='in', length=2)
    plt.savefig('tsotd' + str(ii+1) + '.pdf')


