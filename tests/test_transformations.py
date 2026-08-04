"""Unit tests for transformations module."""

import numpy as np
import pytest

from utils.transformations import (
    geometric_adstock,
    hill_saturation,
    transform_media,
    marginal_response,
    steady_state_adstock,
    effective_spend_range,
)


class TestGeometricAdstock:
    """Tests for geometric_adstock function."""

    def test_basic_adstock(self):
        """Test basic adstock with known values."""
        values = np.array([100.0, 0.0, 0.0, 0.0])
        decay = 0.5
        l_max = 3
        result = geometric_adstock(values, decay, l_max)
        # t=0: 100 * 1 = 100
        # t=1: 100 * 0.5 = 50
        # t=2: 100 * 0.25 = 25
        # t=3: 100 * 0.125 = 12.5
        expected = np.array([100.0, 50.0, 25.0, 12.5])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_zero_decay(self):
        """Test adstock with zero decay (no carryover)."""
        values = np.array([100.0, 200.0, 300.0])
        result = geometric_adstock(values, decay=0.0, l_max=5)
        np.testing.assert_allclose(result, values)

    def test_decay_one(self):
        """Test adstock with decay=1 (permanent carryover)."""
        values = np.array([100.0, 200.0, 300.0])
        result = geometric_adstock(values, decay=1.0, l_max=5)
        # Should be cumulative sum
        expected = np.array([100.0, 300.0, 600.0])
        np.testing.assert_allclose(result, expected)

    def test_empty_array(self):
        """Test adstock with empty array."""
        values = np.array([])
        result = geometric_adstock(values, decay=0.5, l_max=8)
        assert len(result) == 0

    def test_l_max_truncation(self):
        """Test that l_max truncates the carryover."""
        values = np.array([100.0] + [0.0] * 10)
        # With l_max=2, carryover should only go 2 periods
        result = geometric_adstock(values, decay=0.5, l_max=2)
        # t=0: 100, t=1: 50, t=2: 25, t>=3: 0
        assert result[3] == 0.0


class TestHillSaturation:
    """Tests for hill_saturation function."""

    def test_basic_saturation(self):
        """Test basic Hill saturation."""
        values = np.array([0.0, 50.0, 100.0, 1000.0])
        strength = 2.0
        midpoint = 50.0
        result = hill_saturation(values, strength, midpoint)
        # At x=midpoint, result should be 0.5
        np.testing.assert_allclose(result[1], 0.5, rtol=1e-5)
        # At x=0, result should be 0
        assert result[0] == 0.0
        # Should be monotonic increasing
        assert np.all(np.diff(result) >= 0)

    def test_strength_one(self):
        """Test with strength=1 (Michaelis-Menten form)."""
        values = np.array([0.0, 10.0, 100.0])
        strength = 1.0
        midpoint = 50.0
        result = hill_saturation(values, strength, midpoint)
        # x / (x + m)
        expected = values / (values + midpoint)
        np.testing.assert_allclose(result, expected)

    def test_auto_midpoint(self):
        """Test automatic midpoint calculation."""
        values = np.array([10.0, 20.0, 30.0, 40.0])
        result = hill_saturation(values, strength=1.5, midpoint=None)
        # Midpoint should be median of positive values = 25
        # At x=25, result should be 0.5
        idx_mid = np.where(values == 25)[0]
        if len(idx_mid) > 0:
            np.testing.assert_allclose(result[idx_mid[0]], 0.5, rtol=1e-2)

    def test_negative_values(self):
        """Test that negative values are clipped to zero."""
        values = np.array([-10.0, 0.0, 10.0])
        result = hill_saturation(values, strength=1.5, midpoint=10.0)
        # Negative values should be treated as 0
        assert result[0] == 0.0


class TestTransformMedia:
    """Tests for transform_media function."""

    def test_full_pipeline(self):
        """Test full adstock + saturation pipeline."""
        values = np.array([100.0, 0.0, 0.0, 0.0])
        decay = 0.5
        l_max = 3
        strength = 2.0
        midpoint = 50.0
        result = transform_media(values, decay, l_max, strength, midpoint)
        # Should be adstocked then saturated
        adstocked = geometric_adstock(values, decay, l_max)
        expected = hill_saturation(adstocked, strength, midpoint)
        np.testing.assert_allclose(result, expected)


class TestMarginalResponse:
    """Tests for marginal_response function."""

    def test_positive_response(self):
        """Test that marginal response is positive for reasonable inputs."""
        values = np.array([10.0, 50.0, 100.0])
        decay = 0.5
        l_max = 8
        strength = 1.5
        midpoint = 50.0
        result = marginal_response(values, decay, l_max, strength, midpoint)
        assert np.all(result > 0)

    def test_decreasing_marginal(self):
        """Test that marginal response decreases with spend (diminishing returns)."""
        values = np.array([10.0, 50.0, 100.0, 200.0])
        decay = 0.5
        l_max = 8
        strength = 2.0
        midpoint = 50.0
        result = marginal_response(values, decay, l_max, strength, midpoint)
        # Should be decreasing after some point
        assert np.all(np.diff(result) <= 0)


class TestSteadyStateAdstock:
    """Tests for steady_state_adstock function."""

    def test_basic(self):
        """Test steady state calculation."""
        spend = 100.0
        decay = 0.5
        result = steady_state_adstock(spend, decay)
        # 100 / (1 - 0.5) = 200
        np.testing.assert_allclose(result, 200.0, rtol=1e-5)

    def test_zero_decay(self):
        """Test with zero decay."""
        spend = 100.0
        result = steady_state_adstock(spend, decay=0.0)
        np.testing.assert_allclose(result, 100.0, rtol=1e-5)

    def test_high_decay(self):
        """Test with high decay."""
        spend = 100.0
        decay = 0.9
        result = steady_state_adstock(spend, decay)
        # 100 / 0.1 = 1000
        np.testing.assert_allclose(result, 1000.0, rtol=1e-5)


class TestEffectiveSpendRange:
    """Tests for effective_spend_range function."""

    def test_basic(self):
        """Test effective range calculation."""
        decay = 0.5
        l_max = 8
        result = effective_spend_range(decay, l_max)
        # 0.5^0 = 1, 0.5^1 = 0.5, 0.5^2 = 0.25, 0.5^3 = 0.125, 0.5^4 = 0.0625, 0.5^5 = 0.03125
        # 0.5^6 = 0.015625, 0.5^7 = 0.0078125 (< 0.01)
        # So effective range should be 7 (indices 0-6)
        assert result == 7

    def test_high_decay(self):
        """Test with high decay."""
        decay = 0.9
        l_max = 20
        result = effective_spend_range(decay, l_max)
        # 0.9^43 ≈ 0.01, but capped at l_max=20
        # Actually 0.9^k > 0.01 for k up to ~44
        assert result == 20  # capped at l_max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])