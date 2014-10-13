import os
import re

import numpy as np
import pylab as plt


nameconv = {'WMMSE-BS': 'BS', 'WMMSE-D2D': 'D2D',
            'WMMSE-Cell': 'Cellular + D2D'}

plotstyle = {'WMMSE-BS': 'kd-', 'WMMSE-D2D': 'bs-',
             'WMMSE-Cell': 'ro-'}

sims = {}
fls = [f for f in os.listdir('.') if
       re.search('-\d+-\d+-\d+-\d+-\d+-\d+-\d+.npz', f)]

cases = {}
for sim in fls:
    m = re.search('(.*)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+).npz', sim)
    name = m.group(1)
    n_rx = m.group(2)
    n_tx = m.group(3)
    n_ue = m.group(4)
    # n_bs = m.group(5)
    snr = m.group(6)
    radius = m.group(7)
    d2d_dist = m.group(8)

    case = str((n_rx, n_tx, n_ue, snr, d2d_dist))

    if case not in cases:
        cases[case] = {}

    if name not in cases[case]:
        cases[case][name] = {}
        cases[case][name]['x'] = []
        cases[case][name]['y'] = []

    data = np.load(sim)

    cases[case][name]['x'].append(int(radius))
    cases[case][name]['y'].append(data['R'][-1])
    cases[case][name]['reffile'] = sim

i = 1
for case in cases.keys():
    plt.figure(i)
    i += 1

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for name in cases[case].keys():
        plot = cases[case][name]
        plot['x'] = np.array(plot['x'])
        plot['y'] = np.array(plot['y'])

        I = plot['x'].argsort()[::-1]

        plot['x'] = plot['x'][I]
        plot['y'] = plot['y'][I]

        plt.plot(plot['x'], plot['y'], plotstyle[name], label=nameconv[name])

    m = re.search('(.*)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+).npz',
                  cases[case][name]['reffile'])

    name = m.group(1)
    n_rx = m.group(2)
    n_tx = m.group(3)
    n_ue = m.group(4)
    # n_bs = m.group(5)
    snr = m.group(6)
    radius = m.group(7)
    d2d_dist = m.group(8)

    plt.xlabel('Cell radius.')
    plt.ylabel('Average sum rate [bits/sec/Hz].')
    plt.title('$N_{\\mathrm B}=%s, N_{\\mathrm T}=%s, K=%s$, SNR=$%s$dB and D2D distance = $%s$ m' % (n_tx, n_rx, n_ue, snr, d2d_dist))

    legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')

plt.savefig("figure1.png")
plt.show()
