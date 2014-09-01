import logging

# import numpy as np

import utils
import chanmod
import precoder


class Simulator(object):
    def __init__(self, prec, Nt=4, Nr=2, K=2, B=2, SNR=20, realizations=20,
                 biterations=50, channel_model=chanmod.GaussianModel()):

        self.logger = logging.getLogger(__name__)

        self.chanmod = channel_model

        self.Nr = Nr
        self.Nt = Nt
        self.K = K
        self.B = B

        self.realizations = realizations

        self.biterations = biterations

        self.P = 1
        self.N0 = 10**-(SNR/10)

        self.Nsk = min(Nr, Nt)

        if prec is None:
            self.prec = precoder.PrecoderGaussian(Nr, Nt, K, B, self.P, self.N0)

    def iterate_beamformers(self, H):
        (Nr, Nt, K, B, P, N0) = (self.Nr, self.Nt, self.K, self.B, self.P,
                                 self.N0)

        # Initialize beamformers
        rprec = precoder.PrecoderGaussian(Nr, Nt, K, B, P, N0)
        M = rprec.generate()

        for iter in range(self.biterations):
            self.logger.info("Iteration %d/%d" % (iter+1, self.biterations))

            pwr = (M[:]*M[:].conj()).sum()

            M = self.prec.generate()

            C = utils.sigcov(H, M, N0)

            U = utils.lmmse(H, M, N0, C=C)

            E = utils.mse(H, U, M, N0, C=C)

            R = utils.rate(H, M, N0, E=E)

            self.logger.info("Rate: %.2f Power: %.2f%%" %
                             (R[:].sum()/B, 100*(pwr/(B*P))))

    def run(self):
        for rel in range(self.realizations):
            self.logger.info("Realization %d/%d" % (rel+1, self.realizations))

            H = self.chanmod.generate(self.Nr, self.Nt, self.K, self.B)

            self.iterate_beamformers(H)

if __name__ == "__main__":
    sim = Simulator(None)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(levelname)s - %(module)s - %(message)s")
    handler.setFormatter(formatter)

    while (len(logger.handlers) > 0):
        logger.handlers.pop()

    logger.addHandler(handler)

    sim.run()
