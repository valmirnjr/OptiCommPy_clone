import numpy as np
import pytest
import optic.carrierRecovery as carrierRecovery


class TestCarrierRecovery:
    def test_should_generate_bps_test_phases(self):
        π = np.pi

        B = 4
        expected_phases = np.array([(i/B) * (π/2) for i in range(B)])
        test_phases = carrierRecovery.gen_test_phases(B)
        assert np.testing.assert_array_equal(test_phases, expected_phases) is None

        B = 64
        expected_phases = np.array([(i/B) * (π/2) for i in range(B)])
        test_phases = carrierRecovery.gen_test_phases(B)
        assert np.testing.assert_array_equal(test_phases, expected_phases) is None
