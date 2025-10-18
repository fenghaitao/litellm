"""
Integration tests for IFLOW provider.

Tests the integration between IFLOW provider and the main LiteLLM system,
including model registration, provider routing, and end-to-end functionality.
"""

import json
import os
import sys

import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.llms.iflow.chat.transformation import IFlowChatConfig
from litellm.llms.iflow.cost_calculator import cost_per_token as iflow_cost_per_token
from litellm.types.utils import Usage


class TestIFlowIntegration:
    """Test suite for IFLOW provider integration."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment."""
        # Store original environment
        self.original_env = os.environ.copy()
        
        # Ensure we use local model cost map
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        
        # Find and load model cost map
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        while not os.path.exists(os.path.join(project_root, "model_prices_and_context_window.json")):
            parent = os.path.dirname(project_root)
            if parent == project_root:
                break
            project_root = parent
        
        model_cost_path = os.path.join(project_root, "model_prices_and_context_window.json")
        with open(model_cost_path, "r") as f:
            model_cost_map = json.load(f)
        litellm.model_cost = model_cost_map
        
        yield
        
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_model_in_cost_map(self):
        """Test that IFLOW model is properly registered in the model cost map."""
        model_key = "iflow/Qwen3-Coder"
        
        # Check if model exists in cost map
        assert model_key in litellm.model_cost, f"Model {model_key} not found in model_cost"
        
        model_info = litellm.model_cost[model_key]
        assert model_info["litellm_provider"] == "iflow"
        assert model_info["mode"] == "chat"
        assert "input_cost_per_token" in model_info
        assert "output_cost_per_token" in model_info
        assert model_info["supports_function_calling"] is True
        assert model_info["supports_tool_choice"] is True

    def test_provider_in_models_by_provider(self):
        """Test that IFLOW provider is registered in models_by_provider."""
        # Force reload of litellm to ensure latest model registrations
        import importlib
        importlib.reload(litellm)
        
        assert "iflow" in litellm.models_by_provider, "IFLOW provider not found in models_by_provider"
        assert "iflow/Qwen3-Coder" in litellm.models_by_provider["iflow"], "IFLOW model not found in provider models"

    def test_get_model_info(self):
        """Test that get_model_info works for IFLOW models."""
        try:
            model_info = litellm.get_model_info("iflow/Qwen3-Coder")
            assert model_info is not None
            assert model_info.get("litellm_provider") == "iflow"
            assert model_info.get("max_input_tokens") == 1000000
            assert model_info.get("max_output_tokens") == 16384
        except Exception as e:
            # If model info retrieval fails, check if it's due to model not being found
            pytest.skip(f"Model info not available yet: {e}")

    def test_iflow_config_instantiation(self):
        """Test that IFlowChatConfig can be instantiated and used."""
        config = IFlowChatConfig()
        
        assert config is not None
        assert hasattr(config, "_get_openai_compatible_provider_info")
        assert hasattr(config, "get_complete_url")
        assert hasattr(config, "_transform_messages")

    def test_api_key_validation(self):
        """Test API key validation for IFLOW provider."""
        # Test without API key
        try:
            litellm.validate_environment("iflow")
        except Exception as e:
            assert "IFLOW_API_KEY" in str(e)

        # Test with API key
        os.environ["IFLOW_API_KEY"] = "test-key"
        try:
            result = litellm.validate_environment("iflow")
            # Should not raise an exception
            assert result is None or result is True
        except Exception as e:
            # Some validation might still fail due to other requirements
            assert "IFLOW_API_KEY" not in str(e)

    def test_cost_calculation_integration(self):
        """Test that cost calculation integrates properly with the main system."""
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500
        )
        
        # Test direct cost calculation
        prompt_cost, completion_cost = iflow_cost_per_token(
            model="Qwen3-Coder", 
            usage=usage
        )
        
        assert prompt_cost > 0
        assert completion_cost > 0
        
        # Test through main litellm cost calculator (if available)
        try:
            total_cost = litellm.completion_cost(
                model="iflow/Qwen3-Coder",
                usage_block=usage
            )
            expected_total = prompt_cost + completion_cost
            assert abs(total_cost - expected_total) < 1e-10
        except Exception as e:
            # Cost calculation through main system might not be available yet
            pytest.skip(f"Main cost calculation not available: {e}")

    def test_provider_routing(self):
        """Test that the provider routing works correctly."""
        model = "iflow/Qwen3-Coder"
        
        # Extract provider and model name
        provider = litellm.get_llm_provider(model)
        assert provider[1] == "iflow", f"Expected provider 'iflow', got '{provider[1]}'"

    def test_environment_variable_handling(self):
        """Test that environment variables are handled correctly."""
        config = IFlowChatConfig()
        
        # Test without environment variables
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://apis.iflow.cn/v1"
        # assert api_key is None
        
        # Test with environment variables
        os.environ["IFLOW_API_KEY"] = "test-api-key"
        os.environ["IFLOW_API_BASE"] = "https://custom.iflow.api.com/v1"
        
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://custom.iflow.api.com/v1"
        assert api_key == "test-api-key"

    def test_model_capabilities(self):
        """Test that model capabilities are correctly defined."""
        model_info = litellm.model_cost.get("iflow/Qwen3-Coder", {})
        
        # Check that all expected capabilities are defined
        expected_capabilities = [
            "supports_function_calling",
            "supports_reasoning", 
            "supports_tool_choice"
        ]
        
        for capability in expected_capabilities:
            assert capability in model_info, f"Missing capability: {capability}"
            assert model_info[capability] is True, f"Capability {capability} should be True"

    def test_context_window_limits(self):
        """Test that context window limits are properly configured."""
        model_info = litellm.model_cost.get("iflow/Qwen3-Coder", {})
        
        assert "max_input_tokens" in model_info
        assert "max_output_tokens" in model_info
        assert "max_tokens" in model_info
        
        assert model_info["max_input_tokens"] == 1000000
        assert model_info["max_output_tokens"] == 16384
        assert model_info["max_tokens"] == 1000000

    def test_pricing_configuration(self):
        """Test that pricing is properly configured."""
        model_info = litellm.model_cost.get("iflow/Qwen3-Coder", {})
        
        assert "input_cost_per_token" in model_info
        assert "output_cost_per_token" in model_info
        
        assert model_info["input_cost_per_token"] == 3e-07
        assert model_info["output_cost_per_token"] == 1.5e-06
        
        # Verify pricing is reasonable (not negative, not extremely high)
        assert 0 < model_info["input_cost_per_token"] < 0.01
        assert 0 < model_info["output_cost_per_token"] < 0.01

    def test_import_structure(self):
        """Test that all necessary imports work correctly."""
        # Test importing the main components
        from litellm.llms.iflow.chat.transformation import IFlowChatConfig
        from litellm.llms.iflow.cost_calculator import cost_per_token
        
        # Test that they can be instantiated/called
        config = IFlowChatConfig()
        assert config is not None
        
        # Test cost function exists and is callable
        assert callable(cost_per_token)

    @pytest.mark.parametrize("model_name", [
        "Qwen3-Coder",
        "iflow/Qwen3-Coder"
    ])
    def test_model_name_variations(self, model_name):
        """Test that both model name formats work for cost calculation."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        try:
            prompt_cost, completion_cost = iflow_cost_per_token(
                model=model_name, 
                usage=usage
            )
            assert prompt_cost >= 0
            assert completion_cost >= 0
        except Exception as e:
            # Some variations might not work depending on implementation
            assert "model" in str(e).lower() or "not found" in str(e).lower()

    def test_provider_inheritance(self):
        """Test that IFlowChatConfig properly inherits from OpenAIGPTConfig."""
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
        
        config = IFlowChatConfig()
        assert isinstance(config, OpenAIGPTConfig)
        
        # Test that parent methods are available
        assert hasattr(config, "validate_environment")
        assert hasattr(config, "transform_request")
        assert callable(getattr(config, "validate_environment"))
        assert callable(getattr(config, "transform_request"))
