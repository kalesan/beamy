import os
import re

import numpy as np
import pylab as plt

nameconv = {'WMMSE': 'WMMSE', 
            'WMMSE-5': 'WMMSE (5 iterations)', 
            'WMMSE-10': 'WMMSE (10 iterations)', 
            'SDP': 'SDP',
            'SDP-F': 'SDP (Fixed)'}
plotstyle = {'WMMSE': 'k.-', 
            'WMMSE-5': 'b*-', 
            'WMMSE-10': 'gd-', 
            'SDP': 'r*-',
            'SDP-F': 'rs--'}


def plot_sim(filename):
    m = re.search('(.*)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+).npz', sim)
    name = m.group(1)
    n_rx = m.group(2)
    n_tx = m.group(3)
    n_ue = m.group(4)
    # n_bs = m.group(5)
    snr = m.group(6)

    data = np.load(filename)

    niters = data['R'].shape[0]

    if name not in plotstyle:
        return

    plt.plot(range(niters), data['R'], plotstyle[name], label=nameconv[name])
    plt.xlabel('Iteration.')
    plt.ylabel('Average sum rate [bits/sec/Hz].')
    plt.title('$N_{\\mathrm B}=%s, N_{\\mathrm T}=%s, K=%s$ and '
              % (n_tx, n_rx, n_ue) + 'SNR = $%s$dB' % snr)

sims = {}
fls = [f for f in os.listdir('.') if re.search('-\d+-\d+-\d+-\d+-\d+.npz', f)]
for sim in fls:
    m = re.search('(\d+-\d+-\d+-\d+-\d+).npz', sim)
    if m.group(1) not in sims:
        sims[m.group(1)] = [sim]
    else:
        sims[m.group(1)].append(sim)

i = 0
for case in sims.keys():
    plt.figure(i)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    for sim in sims[case]:
        plot_sim(sim)

    legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')

    i += 1
plt.show()
