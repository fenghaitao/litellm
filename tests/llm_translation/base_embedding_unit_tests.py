import asyncio
import httpx
import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import embedding
from litellm.exceptions import BadRequestError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import (
    CustomStreamWrapper,
    get_supported_openai_params,
    get_optional_params,
    get_optional_params_embeddings,
)
import requests
import base64

# test_example.py
from abc import ABC, abstractmethod

url = "https://dummyimage.com/100/100/fff&text=Test+image"
response = requests.get(url)
file_data = response.content

encoded_file = base64.b64encode(file_data).decode("utf-8")
base64_image = f"data:image/png;base64,{encoded_file}"


class BaseLLMEmbeddingTest(ABC):
    """
    Abstract base test class that enforces a common test across all test classes.
    """

    @abstractmethod
    def get_base_embedding_call_args(self) -> dict:
        """Must return the base embedding call args"""
        pass

    @abstractmethod
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        """Must return the custom llm provider"""
        pass

    @pytest.mark.asyncio()
    @pytest.mark.parametrize("sync_mode", [True, False])
    async def test_basic_embedding(self, sync_mode):
        litellm.set_verbose = True
        embedding_call_args = self.get_base_embedding_call_args()
        if sync_mode is True:
            response = litellm.embedding(
                **embedding_call_args,
                input=["hello", "world"],
            )

            print("embedding response: ", response)
        else:
            response = await litellm.aembedding(
                **embedding_call_args,
                input=["hello", "world"],
            )

            print("async embedding response: ", response)

        from openai.types.create_embedding_response import CreateEmbeddingResponse

        CreateEmbeddingResponse.model_validate(response.model_dump())

    def test_embedding_optional_params_max_retries(self):
        embedding_call_args = self.get_base_embedding_call_args()
        optional_params = get_optional_params_embeddings(
            **embedding_call_args, max_retries=20
        )
        assert optional_params["max_retries"] == 20

    def test_image_embedding(self):
        litellm.set_verbose = True
        from litellm.utils import supports_embedding_image_input

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        base_embedding_call_args = self.get_base_embedding_call_args()
        if not supports_embedding_image_input(base_embedding_call_args["model"], None):
            print("Model does not support embedding image input")
            pytest.skip("Model does not support embedding image input")

        embedding(**base_embedding_call_args, input=[base64_image])


class TestGithubCopilotEmbedding(BaseLLMEmbeddingTest):
    """Test GitHub Copilot embedding functionality."""
    
    def get_base_embedding_call_args(self) -> dict:
        return {
            "model": "github_copilot/text-embedding-3-small",
        }
    
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.GITHUB_COPILOT
    
    @pytest.mark.asyncio
    async def test_github_copilot_embedding_with_dimensions(self):
        """Test GitHub Copilot embedding with custom dimensions."""
        litellm.set_verbose = True
        
        # Mock the embedding call since we need credentials for real calls
        with patch('litellm.llms.custom_httpx.llm_http_handler.embedding') as mock_embedding:
            from litellm.types.utils import EmbeddingResponse, Usage
            
            # Mock response with custom dimensions
            mock_response = EmbeddingResponse()
            mock_response.model = "text-embedding-3-small"
            mock_response.data = [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * 512  # Custom 512 dimensions
                }
            ]
            mock_response.usage = Usage(prompt_tokens=4, total_tokens=4)
            mock_response.object = "list"
            
            mock_embedding.return_value = mock_response
            
            response = await litellm.aembedding(
                model="github_copilot/text-embedding-3-small",
                input="Test with custom dimensions",
                dimensions=512
            )
            
            assert response.model == "text-embedding-3-small"
            assert len(response.data) == 1
            assert len(response.data[0]["embedding"]) == 512
            assert response.usage.prompt_tokens == 4
    
    @pytest.mark.asyncio  
    async def test_github_copilot_embedding_with_user_param(self):
        """Test GitHub Copilot embedding with user parameter."""
        litellm.set_verbose = True
        
        with patch('litellm.llms.custom_httpx.llm_http_handler.embedding') as mock_embedding:
            from litellm.types.utils import EmbeddingResponse, Usage
            
            mock_response = EmbeddingResponse()
            mock_response.model = "text-embedding-3-small"
            mock_response.data = [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * 1536
                }
            ]
            mock_response.usage = Usage(prompt_tokens=3, total_tokens=3)
            
            mock_embedding.return_value = mock_response
            
            response = await litellm.aembedding(
                model="github_copilot/text-embedding-3-small",
                input="Test",
                user="test-user-123"
            )
            
            assert response.model == "text-embedding-3-small"
            
            # Verify user parameter was passed
            call_args = mock_embedding.call_args
            assert "user" in call_args[1]["optional_params"]
            assert call_args[1]["optional_params"]["user"] == "test-user-123"
    
    def test_github_copilot_embedding_optional_params(self):
        """Test that GitHub Copilot supports the correct optional parameters."""
        from litellm.utils import get_optional_params_embeddings
        
        optional_params = get_optional_params_embeddings(
            model="github_copilot/text-embedding-3-small",
            custom_llm_provider="github_copilot",
            dimensions=512,
            encoding_format="float",
            user="test-user"
        )
        
        # GitHub Copilot should support these OpenAI-compatible parameters
        assert "dimensions" in optional_params
        assert "encoding_format" in optional_params  
        assert "user" in optional_params
        assert optional_params["dimensions"] == 512
        assert optional_params["encoding_format"] == "float"
        assert optional_params["user"] == "test-user"
    
    def test_github_copilot_embedding_provider_config(self):
        """Test that the provider config is correctly set up."""
        from litellm.utils import ProviderConfigManager
        from litellm.types.utils import LlmProviders
        from litellm.llms.github_copilot.embedding.transformation import GithubCopilotEmbeddingConfig
        
        config = ProviderConfigManager.get_provider_embedding_config(
            model="text-embedding-3-small",
            provider=LlmProviders.GITHUB_COPILOT
        )
        
        assert config is not None
        assert isinstance(config, GithubCopilotEmbeddingConfig)
        
        # Test supported parameters
        supported_params = config.get_supported_openai_params("text-embedding-3-small")
        expected_params = ["encoding_format", "dimensions", "user"]
        
        for param in expected_params:
            assert param in supported_params
