""" System model describes the essential system parameters for the simulator. It
    contains the channel model and gain model as well as the SNR settings. """

import chanmod
import gainmod


class SystemModel(object):

    """SystemModel is a container class for the essential system parameters."""

    def __init__(self, dx, bx, K, B, **kwargs):
        """

        Args:
            dx (int): device antennas of receive antennas.
            bx (int): number of base station antennas.
            K (int): number of mobile devices (or device pairs).
            B (int): number of base stations.

        Kwargs:
            SNR: Cell edge sum SNR [dB].
            gainmod: Channel gain model.
            chanmod: Channel model.
        """

        # Common system parameters in a tupple
        self.sysparams = (dx, bx, K, B)

        self.SNR = kwargs.get('SNR', 20)

        self.gainmod = kwargs.get('gainmod', gainmod.Uniform1(K, B,
                                                              SNR=self.SNR))

        self.chanmod = kwargs.get('chanmod', chanmod.GaussianModel())

        self.chan = {}
