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
    uDeep.append(data[:,ndim+2:2*ndim+2])
    uNum.append(data[:,2*ndim+2::])

colors = ["#1b9e77", "#d95f02", "#7570b3"]
ls = ["-", "-", "--"]
ind_st = 2000
xticks = [time[0][ind_st]+2*ii*np.pi for ii in range(3)]
yticks = [1e-10, 1e-6, 1e-2]

fig = plt.figure(figsize=(1.7,1.6), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0, h_pad=0.02)
ax = plt.axes()
for ii in range(notd):
    dist = 1-np.abs(np.einsum('ij,ij->i', uDeep[ii], uNum[ii]))
    plt.semilogy(time[ii], dist, color=colors[ii], linewidth=0.75, ls=ls[ii])
plt.xlabel(r'$t-{0:0.0f}$'.format(time[0][ind_st]))
plt.ylabel(r'$d_i(\mathbf{x}(t))$')
plt.xlim(xticks[0], xticks[-1])
plt.ylim(yticks[0], yticks[-1])
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels(latexify(["0", "2\pi", "4\pi"]))
ax.set_yticklabels(latexify(["10^{-10}", "10^{-6}", "10^{-2}"]))
ax.tick_params(direction='in', length=2)
plt.savefig('dist_3dim.pdf')


