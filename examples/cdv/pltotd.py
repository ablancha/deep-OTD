import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def latexify(ticklabels):
    """Manually set LaTeX format for tick labels."""
    return [r"$" + str(label) + "$" for label in ticklabels]
         
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 9

notd = 1
ndim = 6

time = []
traj = []
uDeep = []
uNum = []

for ii in range(notd):
    data = np.genfromtxt('dOTD_tst' + str(ii+1) + '.out')
    time.append(data[:,0])
    traj.append(data[:,1:ndim+1])
    uDeep.append(data[:,ndim+1:2*ndim+1])
    uNum.append(data[:,2*ndim+1::])

ind_st = 1300
xticks = np.linspace(500, 4000, 8, dtype=int).tolist()
yticks = [-1.2, 0, 1.2]

for ii in range(notd):
    fig = plt.figure(figsize=(6.7,7), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0.1)
    for jj in range(ndim):
        ax = plt.subplot(ndim,1,jj+1)
        plt.plot(time[ii], uDeep[ii][:,jj], color="#ca0020", linewidth=0.75)
        plt.plot(time[ii], uNum[ii][:,jj], color="#0571b0", linewidth=0.75)
        ax.axvspan(1075, 1165, color='0.9')
        plt.xlabel(r'$t$')
        plt.ylabel('$u_{' + str(ii+1) + ',' + 'z_{' + str(jj+1) + '}}$')
        plt.xlim(xticks[0], xticks[-1])
        plt.ylim(yticks[0], yticks[-1])
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(latexify(xticks))
        ax.set_yticklabels(latexify(yticks))
        ax.tick_params(direction='in', length=2)
    plt.savefig('tsotd' + str(ii+1) + '.pdf')


