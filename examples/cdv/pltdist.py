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
    uDeep.append(data[:,ndim+1:2*ndim+1])
    uNum.append(data[:,2*ndim+1::])

xticks = np.linspace(500, 4000, 8, dtype=int).tolist()
yticks_traj = [0.7, 1]
yticks_dist = [0, 0.5, 1]

figsize = (6.7,1.1)
labelx = -0.05

fig = plt.figure(figsize=figsize, constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
ax = plt.axes()
plt.plot(data[:,0], data[:,1], 'k-', linewidth=0.75)
ax.axvspan(1075, 1165, color='0.9')
plt.xlabel('$t$')
plt.ylabel('$z_1$')
plt.xlim(xticks[0], xticks[-1])
plt.ylim(yticks_traj[0], yticks_traj[-1])
ax.set_xticks(xticks)
ax.set_yticks(yticks_traj)
ax.set_xticklabels(latexify(xticks))
ax.set_yticklabels(latexify(yticks_traj))
ax.yaxis.set_label_coords(labelx, 0.5)
ax.tick_params(direction='in', length=2)
plt.savefig('dist_z1.pdf')


for ii in range(notd):

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
    ax = plt.axes()
    dist = 1-np.abs(np.einsum('ij,ij->i', uDeep[ii], uNum[ii]))
    plt.plot(time[ii], dist, color='#ca0020', linewidth=0.75, zorder=10)
    ax.axvspan(1075, 1165, color='0.9')
    plt.xlabel('$t$')
    plt.ylabel(r'$d_{' + str(ii+1) + '}(\mathbf{x}(t))$')
    plt.xlim(xticks[0], xticks[-1])
    plt.ylim(yticks_dist[0], yticks_dist[-1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks_dist)
    ax.set_xticklabels(latexify(xticks))
    ax.set_yticklabels(latexify(yticks_dist))
    ax.yaxis.set_label_coords(labelx, 0.5)
    ax.tick_params(direction='in', length=2)
    plt.savefig('dist' + str(ii+1) + '.pdf')






