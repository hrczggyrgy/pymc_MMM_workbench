"""Unit tests for optimization module."""

import numpy as np
import pytest
import pandas as pd

from utils.optimization import (
    expected_response,
    optimize_budget,
    compute_channel_roi,
    response_curve_data,
)
from utils.simulation import generate_demo_data
from utils.modeling import fit_bayesian_mmm


class MockResult:
    """Mock model result for testing optimization functions."""
    
    def __init__(self):
        self.channels = ["search", "social", "video"]
        self.current_spend = {"search": 10000.0, "social": 5000.0, "video": 8000.0}
        self.features = ["search", "social", "video", "price", "promo", "trend", "sin_annual", "cos_annual"]
        self.y_std = 1000.0
        
        # Mock params
        self.params = {
            "search": {"decay": 0.5, "strength": 1.5, "midpoint": 5000.0},
            "social": {"decay": 0.4, "strength": 1.8, "midpoint": 3000.0},
            "video": {"decay": 0.6, "strength": 1.2, "midpoint": 6000.0},
        }
        
        # Mock beta draws (n_draws, n_features)
        np.random.seed(42)
        n_draws = 100
        n_features = len(self.features)
        self.beta_draws = np.random.normal(0, 0.5, (n_draws, n_features))
        # Make channel betas positive
        for i, ch in enumerate(self.channels):
            self.beta_draws[:, i] = np.abs(self.beta_draws[:, i]) + 0.1
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __contains__(self, key):
        return hasattr(self, key)


class TestExpectedResponse:
    """Tests for expected_response function."""
    
    def test_positive_response(self):
        """Test that expected response is positive for positive spend."""
        mock = MockResult()
        spend = 10000.0
        beta_draws = mock.beta_draws[:, 0]  # search channel
        decay = 0.5
        strength = 1.5
        midpoint = 5000.0
        y_std = 1000.0
        
        resp = expected_response(spend, beta_draws, decay, strength, midpoint, y_std)
        assert np.all(resp > 0)
        assert resp.shape == beta_draws.shape
    
    def test_zero_spend(self):
        """Test that zero spend gives zero response."""
        mock = MockResult()
        spend = 0.0
        beta_draws = mock.beta_draws[:, 0]
        decay = 0.5
        strength = 1.5
        midpoint = 5000.0
        y_std = 1000.0
        
        resp = expected_response(spend, beta_draws, decay, strength, midpoint, y_std)
        assert np.all(resp == 0.0)


class TestOptimizeBudget:
    """Tests for optimize_budget function."""
    
    def test_budget_constraint(self):
        """Test that optimization respects budget constraint."""
        mock = MockResult()
        total_budget = 23000.0  # Same as current total
        minimums = {c: 0.0 for c in mock.channels}
        maximums = {c: 50000.0 for c in mock.channels}
        
        allocation, lift = optimize_budget(total_budget, minimums, maximums, mock, n_draws=50)
        
        # Check budget is respected
        assert abs(sum(allocation.values()) - total_budget) < 1.0
        
        # Check bounds
        for c in mock.channels:
            assert minimums[c] <= allocation[c] <= maximums[c]
    
    def test_minimum_bounds(self):
        """Test that minimum bounds are respected."""
        mock = MockResult()
        total_budget = 20000.0
        minimums = {"search": 8000.0, "social": 3000.0, "video": 5000.0}
        maximums = {c: 50000.0 for c in mock.channels}
        
        allocation, lift = optimize_budget(total_budget, minimums, maximums, mock, n_draws=50)
        
        for c in mock.channels:
            assert allocation[c] >= minimums[c] - 1e-6
    
    def test_maximum_bounds(self):
        """Test that maximum bounds are respected."""
        mock = MockResult()
        total_budget = 20000.0
        minimums = {c: 0.0 for c in mock.channels}
        maximums = {"search": 12000.0, "social": 6000.0, "video": 9000.0}
        
        allocation, lift = optimize_budget(total_budget, minimums, maximums, mock, n_draws=50)
        
        for c in mock.channels:
            assert allocation[c] <= maximums[c] + 1e-6
    
    def test_infeasible_minimums(self):
        """Test that infeasible minimums raise error."""
        mock = MockResult()
        total_budget = 10000.0
        minimums = {c: 5000.0 for c in mock.channels}  # Sum = 15000 > 10000
        maximums = {c: 50000.0 for c in mock.channels}
        
        with pytest.raises(ValueError, match="exceeds total budget"):
            optimize_budget(total_budget, minimums, maximums, mock, n_draws=50)
    
    def test_infeasible_maximums(self):
        """Test that infeasible maximums raise error."""
        mock = MockResult()
        total_budget = 30000.0
        minimums = {c: 0.0 for c in mock.channels}
        maximums = {c: 5000.0 for c in mock.channels}  # Sum = 15000 < 30000
        
        with pytest.raises(ValueError, match="less than total budget"):
            optimize_budget(total_budget, minimums, maximums, mock, n_draws=50)
    
    def test_lift_samples_shape(self):
        """Test that lift samples have correct shape."""
        mock = MockResult()
        total_budget = 23000.0
        minimums = {c: 0.0 for c in mock.channels}
        maximums = {c: 50000.0 for c in mock.channels}
        
        allocation, lift = optimize_budget(total_budget, minimums, maximums, mock, n_draws=50)
        
        assert lift.shape == (50,)


class TestComputeChannelROI:
    """Tests for compute_channel_roi function."""
    
    def test_roi_positive(self):
        """Test that ROI is positive for positive response."""
        mock = MockResult()
        allocation = {"search": 10000.0, "social": 5000.0, "video": 8000.0}
        
        roi_df = compute_channel_roi(allocation, mock, n_draws=50)
        
        assert len(roi_df) == 3
        assert all(roi_df["ROI"] > 0)
        assert all(roi_df["Marginal_ROAS"] > 0)
        assert all(roi_df["Spend"] > 0)
    
    def test_roi_columns(self):
        """Test that all expected columns are present."""
        mock = MockResult()
        allocation = {"search": 10000.0, "social": 5000.0, "video": 8000.0}
        
        roi_df = compute_channel_roi(allocation, mock, n_draws=50)
        
        expected_cols = ["Channel", "Spend", "Expected_Response", "ROI", "Marginal_ROAS"]
        assert list(roi_df.columns) == expected_cols


class TestResponseCurveData:
    """Tests for response_curve_data function."""
    
    def test_curve_shape(self):
        """Test that response curve has correct shape."""
        mock = MockResult()
        
        spend_vals, mean_resp, low_resp, high_resp = response_curve_data("search", mock, max_multiplier=3.0, n_points=100)
        
        assert len(spend_vals) == 100
        assert len(mean_resp) == 100
        assert len(low_resp) == 100
        assert len(high_resp) == 100
    
    def test_curve_monotonic(self):
        """Test that response curve is monotonic increasing."""
        mock = MockResult()
        
        spend_vals, mean_resp, low_resp, high_resp = response_curve_data("search", mock, max_multiplier=3.0, n_points=100)
        
        assert np.all(np.diff(mean_resp) >= -1e-6)  # Allow tiny numerical errors
    
    def test_credible_intervals(self):
        """Test that credible intervals are ordered correctly."""
        mock = MockResult()
        
        spend_vals, mean_resp, low_resp, high_resp = response_curve_data("search", mock, max_multiplier=3.0, n_points=100)
        
        assert np.all(low_resp <= mean_resp)
        assert np.all(mean_resp <= high_resp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])