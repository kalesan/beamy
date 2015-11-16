import os
import re

import numpy as np

nameconv = {'WMMSE-MAC': 'WMMSE', 
            'SDP-MAC': 'SDP'}

def plot_sim(filename, first):
    m = re.search('(.*)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+).npz', sim)
    name = m.group(1)
    n_rx = m.group(2)
    n_tx = m.group(3)
    n_ue = m.group(4)
    # n_bs = m.group(5)
    snr = m.group(6)

    data = np.load(filename)

    niters = data['R'].shape[0]

    if name not in nameconv:
        return

    if first:
        print
        print
        print('Tx=%s, Rx=%s, K=%s and '
                % (n_tx, n_rx, n_ue) + 'SNR = %sdB' % snr)

    print
    print('%s: asymptotic performance %f' % (name, data['R'][-1]))
    print('%s: average number of iterations %d' % (name, data['I']))

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
    first = True 

    for sim in sims[case]:
        plot_sim(sim, first)
        first = False

    i += 1
