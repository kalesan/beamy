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
    n_ue = float(m.group(4))
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
        cases[case][name]['R'] = []
        cases[case][name]['S_B2D'] = []
        cases[case][name]['S_D2B'] = []
        cases[case][name]['S_D2D_1'] = []
        cases[case][name]['S_D2D_2'] = []

    data = np.load(sim)

    cases[case][name]['x'].append(int(radius))
    cases[case][name]['R'].append(data['R'][-1] / n_ue)
    cases[case][name]['S_B2D'].append(data['S_B2D'][-1] / n_ue)
    cases[case][name]['S_D2B'].append(data['S_D2B'][-1] / n_ue)
    cases[case][name]['S_D2D_1'].append(data['S_D2D_1'][-1] / n_ue)
    cases[case][name]['S_D2D_2'].append(data['S_D2D_2'][-1] / n_ue)
    cases[case][name]['reffile'] = sim

i = 1
# Rate
for case in cases.keys():
    plt.figure(i)
    i += 1

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for name in cases[case].keys():
        plot = cases[case][name]
        plot['x'] = np.array(plot['x'])
        plot['R'] = np.array(plot['R'])

        I = plot['x'].argsort()[::-1]

        x = plot['x'][I].copy()
        R = plot['R'][I].copy()

        plt.plot(x, R, plotstyle[name], label=nameconv[name])

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

# Stream Allocation
for case in cases.keys():
    plt.figure(i)
    i += 1

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for name in cases[case].keys():
        if name != "WMMSE-Cell":
            continue

        plot = cases[case][name]

        x = np.array(plot['x'])

        I = plot['x'].argsort()[::-1]

        R = np.array(plot['R'])
        S_B2D = np.array(plot['S_B2D'])
        S_D2B = np.array(plot['S_D2B'])
        S_D2D_1 = np.array(plot['S_D2D_1'])
        S_D2D_2 = np.array(plot['S_D2D_2'])

        S_plt_style = {'S_B2D': 'kd-', 'S_D2B': 'bs-', 'S_D2D_1': 'ro-',
                       'S_D2D_2': 'rs--'}

        plt.plot(x[I], S_B2D[I], S_plt_style['S_B2D'], label='B2D')
        plt.plot(x[I], S_D2B[I], S_plt_style['S_D2B'], label='D2B')
        plt.plot(x[I], S_D2D_1[I], S_plt_style['S_D2D_1'], label='D2D (1)')
        plt.plot(x[I], S_D2D_2[I], S_plt_style['S_D2D_2'], label='D2D (2)')

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
    plt.ylabel('Average stream allocation [bits/sec/Hz].')
    plt.title('$N_{\\mathrm B}=%s, N_{\\mathrm T}=%s, K=%s$, SNR=$%s$dB and D2D distance = $%s$ m' % (n_tx, n_rx, n_ue, snr, d2d_dist))

    legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.show()
