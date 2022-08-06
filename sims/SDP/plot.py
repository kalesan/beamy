import numpy as np
import pylab as plt

dwmmse = np.load('wmmse.npz')
dsdp = np.load('sdp.npz')

niters = dwmmse['R'].shape[0]

plt.plot(range(niters), dwmmse['R'], 'k--', label='WMMSE')
plt.plot(range(niters), dsdp['R'], 'r-', label='SDP')

legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')

# legend.get_frame().set_facecolor('#00FFCC')

plt.show()
