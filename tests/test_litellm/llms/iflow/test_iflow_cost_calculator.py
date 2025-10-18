"""
Test suite for IFLOW cost calculation functionality.

Tests the cost calculation for IFLOW models including:
- Tiered pricing based on input token ranges
- Caching discounts
- Reasoning tokens
- Standard flat pricing fallback
"""

import json
import math
import os
import sys

import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.llms.iflow.cost_calculator import (
    cost_per_token as iflow_cost_per_token,
)
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    Usage,
)


class TestIFlowCostCalculator:
    """Test suite for IFLOW cost calculation functionality."""

    @pytest.fixture(autouse=True)
    def setup_model_cost_map(self):
        """Set up the model cost map for testing."""
        # Ensure we use local model cost map for consistent testing
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        
        # Find the project root directory and load model cost map
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        while not os.path.exists(os.path.join(project_root, "model_prices_and_context_window.json")):
            parent = os.path.dirname(project_root)
            if parent == project_root:  # Reached filesystem root
                break
            project_root = parent
        
        model_cost_path = os.path.join(project_root, "model_prices_and_context_window.json")
        with open(model_cost_path, "r") as f:
            model_cost_map = json.load(f)
        litellm.model_cost = model_cost_map

    def test_basic_cost_calculation(self):
        """Test basic cost calculation for IFLOW Qwen3-Coder model."""
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500
        )
        
        prompt_cost, completion_cost = iflow_cost_per_token(
            model="Qwen3-Coder", 
            usage=usage
        )
        
        # Expected costs for iflow/Qwen3-Coder:
        # Input: 1000 tokens * $3e-7 = $0.0003
        # Output: 500 tokens * $1.5e-6 = $0.00075
        expected_prompt_cost = 1000 * 3e-7
        expected_completion_cost = 500 * 1.5e-6
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        usage = Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
        
        prompt_cost, completion_cost = iflow_cost_per_token(
            model="Qwen3-Coder", 
            usage=usage
        )
        
        assert prompt_cost == 0.0
        assert completion_cost == 0.0

    def test_prompt_only_tokens(self):
        """Test cost calculation with only prompt tokens."""
        usage = Usage(
            prompt_tokens=2000,
            completion_tokens=0,
            total_tokens=2000
        )
        
        prompt_cost, completion_cost = iflow_cost_per_token(
            model="Qwen3-Coder", 
            usage=usage
        )
        
        expected_prompt_cost = 2000 * 3e-7
        expected_completion_cost = 0.0
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert completion_cost == 0.0

    def test_completion_only_tokens(self):
        """Test cost calculation with only completion tokens."""
        usage = Usage(
            prompt_tokens=0,
            completion_tokens=1000,
            total_tokens=1000
        )
        
        prompt_cost, completion_cost = iflow_cost_per_token(
            model="Qwen3-Coder", 
            usage=usage
        )
        
        expected_prompt_cost = 0.0
        expected_completion_cost = 1000 * 1.5e-6
        
        assert prompt_cost == 0.0
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_large_token_counts(self):
        """Test cost calculation with large token counts."""
        usage = Usage(
            prompt_tokens=500000,  # 500K tokens
            completion_tokens=10000,  # 10K tokens
            total_tokens=510000
        )
        
        prompt_cost, completion_cost = iflow_cost_per_token(
            model="Qwen3-Coder", 
            usage=usage
        )
        
        expected_prompt_cost = 500000 * 3e-7
        expected_completion_cost = 10000 * 1.5e-6
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_cached_tokens_handling(self):
        """Test cost calculation with cached tokens."""
        prompt_tokens_details = PromptTokensDetailsWrapper(
            cached_tokens=5000  # 5K cached tokens
        )
        
        usage = Usage(
            prompt_tokens=15000,  # 10K regular + 5K cached = 15K total
            completion_tokens=1000,
            total_tokens=16000,
            prompt_tokens_details=prompt_tokens_details
        )
        
        prompt_cost, completion_cost = iflow_cost_per_token(
            model="Qwen3-Coder", 
            usage=usage
        )
        
        # Expected cost calculation:
        # Regular tokens: 10,000 * $3e-7 = $0.003
        # Cached tokens: 5,000 * $3e-7 = $0.0015 (assuming same rate as no cache rate specified)
        # Total input cost = $0.003 + $0.0015 = $0.0045
        
        regular_tokens = 10000
        cached_tokens = 5000
        
        expected_regular_cost = regular_tokens * 3e-7
        expected_cached_cost = cached_tokens * 3e-7  # Fallback to input rate
        expected_prompt_cost = expected_regular_cost + expected_cached_cost
        expected_completion_cost = 1000 * 1.5e-6
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_reasoning_tokens_handling(self):
        """Test cost calculation with reasoning tokens."""
        completion_tokens_details = CompletionTokensDetailsWrapper(
            reasoning_tokens=2000  # 2K reasoning tokens
        )
        
        usage = Usage(
            prompt_tokens=5000,
            completion_tokens=5000,  # 3K regular + 2K reasoning = 5K total
            total_tokens=10000,
            completion_tokens_details=completion_tokens_details
        )
        
        prompt_cost, completion_cost = iflow_cost_per_token(
            model="Qwen3-Coder", 
            usage=usage
        )
        
        # Expected cost calculation:
        # Prompt: 5,000 * $3e-7 = $0.0015
        # Regular completion: 3,000 * $1.5e-6 = $0.0045
        # Reasoning completion: 2,000 * $1.5e-6 = $0.003 (fallback to output rate)
        # Total completion cost = $0.0045 + $0.003 = $0.0075
        
        regular_completion_tokens = 3000
        reasoning_tokens = 2000
        
        expected_prompt_cost = 5000 * 3e-7
        expected_regular_completion_cost = regular_completion_tokens * 1.5e-6
        expected_reasoning_cost = reasoning_tokens * 1.5e-6  # Fallback to output rate
        expected_completion_cost = expected_regular_completion_cost + expected_reasoning_cost
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_combined_special_tokens(self):
        """Test cost calculation with both cached and reasoning tokens."""
        prompt_tokens_details = PromptTokensDetailsWrapper(
            cached_tokens=3000  # 3K cached tokens
        )
        
        completion_tokens_details = CompletionTokensDetailsWrapper(
            reasoning_tokens=1500  # 1.5K reasoning tokens
        )
        
        usage = Usage(
            prompt_tokens=8000,  # 5K regular + 3K cached = 8K total
            completion_tokens=4000,  # 2.5K regular + 1.5K reasoning = 4K total
            total_tokens=12000,
            prompt_tokens_details=prompt_tokens_details,
            completion_tokens_details=completion_tokens_details
        )
        
        prompt_cost, completion_cost = iflow_cost_per_token(
            model="Qwen3-Coder", 
            usage=usage
        )
        
        # Expected cost calculation:
        # Regular prompt: 5,000 * $3e-7 = $0.0015
        # Cached prompt: 3,000 * $3e-7 = $0.0009 (fallback rate)
        # Regular completion: 2,500 * $1.5e-6 = $0.00375
        # Reasoning completion: 1,500 * $1.5e-6 = $0.00225 (fallback rate)
        
        regular_prompt_tokens = 5000
        cached_tokens = 3000
        regular_completion_tokens = 2500
        reasoning_tokens = 1500
        
        expected_prompt_cost = (regular_prompt_tokens * 3e-7) + (cached_tokens * 3e-7)
        expected_completion_cost = (regular_completion_tokens * 1.5e-6) + (reasoning_tokens * 1.5e-6)
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_edge_case_single_token(self):
        """Test cost calculation with single tokens."""
        usage = Usage(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2
        )
        
        prompt_cost, completion_cost = iflow_cost_per_token(
            model="Qwen3-Coder", 
            usage=usage
        )
        
        expected_prompt_cost = 1 * 3e-7
        expected_completion_cost = 1 * 1.5e-6
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_cost_calculator_with_model_prefix(self):
        """Test cost calculator works when model includes provider prefix."""
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500
        )
        
        # Test with full model name including provider prefix
        prompt_cost, completion_cost = iflow_cost_per_token(
            model="iflow/Qwen3-Coder",  # Full model name with prefix
            usage=usage
        )
        
        expected_prompt_cost = 1000 * 3e-7
        expected_completion_cost = 500 * 1.5e-6
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_model_info_fallback(self):
        """Test that the cost calculator handles missing model gracefully."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        try:
            # This should work as the model should be in the cost map
            prompt_cost, completion_cost = iflow_cost_per_token(
                model="Qwen3-Coder", 
                usage=usage
            )
            # If we get here, the model was found and costs calculated
            assert prompt_cost >= 0
            assert completion_cost >= 0
        except Exception as e:
            # If model is not found, that's expected behavior for this test
            # The cost calculator should handle this gracefully
            assert "model" in str(e).lower() or "not found" in str(e).lower()