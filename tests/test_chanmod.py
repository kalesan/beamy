import pytest
import numpy as np

from beamy import chanmod

class TestGaussianModel:
    def test_variance(self):
        """Test that Gaussian channel model has unit variance"""  
        model = chanmod.GaussianModel((1000, 100, 3, 3))

        chan = model.generate()

        for ue in range(3):
            for bs in range(3):
                np.testing.assert_almost_equal(np.var(chan[:, :, ue, bs][:]), 1, 1e-3)

