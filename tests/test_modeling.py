"""Smoke tests for modeling module."""

import numpy as np
import pytest

from utils.simulation import generate_demo_data
from utils.modeling import fit_bayesian_mmm


class TestModelingSmoke:
    """Smoke tests for model fitting - fast mode with minimal draws."""
    
    def test_fit_demo_data(self):
        """Test that model fits demo data without errors."""
        df = generate_demo_data(n_periods=52, seed=42)
        channels = ["search", "social", "video", "display"]
        controls = ["price", "promo"]
        
        decay = {"search": 0.58, "social": 0.42, "video": 0.72, "display": 0.30}
        strength = {"search": 1.35, "social": 1.7, "video": 1.2, "display": 1.9}
        
        result = fit_bayesian_mmm(
            df, "date", "sales",
            channels, controls, decay, 8, strength, 1.0,
            True, 1.0, draws=20, tune=20, seed=42
        )
        
        # Check result structure
        assert "idata" in result
        assert "prediction" in result
        assert "summary" in result
        assert "beta_draws" in result
        assert "params" in result
        assert "channels" in result
        assert "current_spend" in result
        
        # Check channels
        assert result["channels"] == channels
        
        # Check beta draws shape
        assert result["beta_draws"].shape[1] == len(result["features"])
        assert result["beta_draws"].shape[0] > 0
        
        # Check prediction shape
        assert len(result["prediction"]) == 52
        assert "mean" in result["prediction"].columns
        assert "low" in result["prediction"].columns
        assert "high" in result["prediction"].columns
    
    def test_fit_with_test_size(self):
        """Test that model works with holdout validation."""
        df = generate_demo_data(n_periods=52, seed=42)
        channels = ["search", "social", "video", "display"]
        controls = ["price", "promo"]
        
        decay = {"search": 0.58, "social": 0.42, "video": 0.72, "display": 0.30}
        strength = {"search": 1.35, "social": 1.7, "video": 1.2, "display": 1.9}
        
        result = fit_bayesian_mmm(
            df, "date", "sales",
            channels, controls, decay, 8, strength, 1.0,
            True, 1.0, draws=20, tune=20, seed=42, test_size=0.2
        )
        
        assert "test_prediction" in result
        assert "test_metrics" in result
        assert result["test_prediction"] is not None
        assert result["test_metrics"] is not None
        assert result["test_n"] == 10  # 20% of 52
        assert result["train_n"] == 42
        
        metrics = result["test_metrics"]
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "r2" in metrics
        assert "n_test" in metrics
    
    def test_per_channel_params(self):
        """Test that per-channel decay/strength are stored correctly."""
        df = generate_demo_data(n_periods=52, seed=42)
        channels = ["search", "social", "video", "display"]
        controls = ["price", "promo"]
        
        decay = {"search": 0.58, "social": 0.42, "video": 0.72, "display": 0.30}
        strength = {"search": 1.35, "social": 1.7, "video": 1.2, "display": 1.9}
        
        result = fit_bayesian_mmm(
            df, "date", "sales",
            channels, controls, decay, 8, strength, 1.0,
            True, 1.0, draws=20, tune=20, seed=42
        )
        
        # Check params stored per channel
        for c in channels:
            assert c in result["params"]
            assert "decay" in result["params"][c]
            assert "strength" in result["params"][c]
            assert result["params"][c]["decay"] == decay[c]
            assert result["params"][c]["strength"] == strength[c]


class TestScenarioLift:
    """Tests for scenario_lift function."""
    
    def test_scenario_lift_basic(self):
        """Test that scenario_lift returns correct shape."""
        df = generate_demo_data(n_periods=52, seed=42)
        channels = ["search", "social", "video", "display"]
        controls = ["price", "promo"]
        
        decay = {"search": 0.58, "social": 0.42, "video": 0.72, "display": 0.30}
        strength = {"search": 1.35, "social": 1.7, "video": 1.2, "display": 1.9}
        
        result = fit_bayesian_mmm(
            df, "date", "sales",
            channels, controls, decay, 8, strength, 1.0,
            True, 1.0, draws=20, tune=20, seed=42
        )
        
        from utils.modeling import scenario_lift
        
        # Test 10% increase
        plan = {c: v * 1.1 for c, v in result["current_spend"].items()}
        lift = scenario_lift(plan, result)
        
        assert len(lift) > 0
        assert np.isfinite(lift).all()


class TestResponseSamples:
    """Tests for response_samples function."""
    
    def test_response_samples_shape(self):
        """Test that response_samples returns correct shape."""
        df = generate_demo_data(n_periods=52, seed=42)
        channels = ["search", "social", "video", "display"]
        controls = ["price", "promo"]
        
        decay = {"search": 0.58, "social": 0.42, "video": 0.72, "display": 0.30}
        strength = {"search": 1.35, "social": 1.7, "video": 1.2, "display": 1.9}
        
        result = fit_bayesian_mmm(
            df, "date", "sales",
            channels, controls, decay, 8, strength, 1.0,
            True, 1.0, draws=20, tune=20, seed=42
        )
        
        from utils.modeling import response_samples
        
        spend = 10000.0
        # With draws=20, tune=20, we get 40 posterior samples (2 chains * 20 draws)
        # response_samples clips n to min(n, len(beta_draws))
        resp = response_samples(spend, "search", result, n=50)
        
        # Should be clipped to available samples
        assert len(resp) <= 50
        assert np.isfinite(resp).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])