"""
Test module for GitHub Copilot embedding functionality in litellm.

This module contains comprehensive tests for the GitHub Copilot embedding implementation,
including unit tests for configuration, transformation logic, and integration tests.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
import httpx

from litellm.llms.github_copilot.embedding.transformation import GithubCopilotEmbeddingConfig
from litellm.llms.github_copilot.common_utils import GithubCopilotError, GetAPIKeyError
from litellm.types.utils import EmbeddingResponse, Usage
from litellm.exceptions import AuthenticationError


class TestGithubCopilotEmbeddingConfig:
    """Test the GitHub Copilot embedding configuration class."""
    
    @pytest.fixture
    def embedding_config(self):
        """Create a GithubCopilotEmbeddingConfig instance for testing."""
        return GithubCopilotEmbeddingConfig()
    
    @pytest.fixture
    def mock_authenticator(self):
        """Create a mock authenticator."""
        with patch('litellm.llms.github_copilot.embedding.transformation.Authenticator') as mock_auth:
            mock_auth_instance = mock_auth.return_value
            mock_auth_instance.get_api_key.return_value = "test-api-key"
            mock_auth_instance.get_api_base.return_value = None
            yield mock_auth_instance
    
    def test_initialization(self, embedding_config):
        """Test that the embedding config initializes correctly."""
        assert embedding_config is not None
        assert hasattr(embedding_config, 'authenticator')
        assert embedding_config.GITHUB_COPILOT_API_BASE == "https://api.githubcopilot.com/"
    
    def test_get_supported_openai_params(self, embedding_config):
        """Test that supported OpenAI parameters are correctly defined."""
        params = embedding_config.get_supported_openai_params("text-embedding-3-small")
        expected_params = ["encoding_format", "dimensions", "user"]
        
        assert isinstance(params, list)
        for param in expected_params:
            assert param in params
    
    def test_get_complete_url_default(self, embedding_config, mock_authenticator):
        """Test URL generation with default API base."""
        url = embedding_config.get_complete_url(
            api_base=None,
            api_key=None,
            model="text-embedding-3-small",
            optional_params={},
            litellm_params={}
        )
        
        assert url == "https://api.githubcopilot.com/embeddings"
    
    def test_get_complete_url_custom_base(self, embedding_config, mock_authenticator):
        """Test URL generation with custom API base."""
        custom_base = "https://api.custom.githubcopilot.com"
        url = embedding_config.get_complete_url(
            api_base=custom_base,
            api_key=None,
            model="text-embedding-3-small",
            optional_params={},
            litellm_params={}
        )
        
        assert url == f"{custom_base}/embeddings"
    
    def test_get_complete_url_with_trailing_slash(self, embedding_config, mock_authenticator):
        """Test URL generation removes trailing slashes correctly."""
        custom_base = "https://api.custom.githubcopilot.com/"
        url = embedding_config.get_complete_url(
            api_base=custom_base,
            api_key=None,
            model="text-embedding-3-small",
            optional_params={},
            litellm_params={}
        )
        
        assert url == "https://api.custom.githubcopilot.com/embeddings"
    
    def test_get_complete_url_with_existing_embeddings_path(self, embedding_config, mock_authenticator):
        """Test URL generation when path already contains embeddings."""
        custom_base = "https://api.custom.githubcopilot.com/embeddings"
        url = embedding_config.get_complete_url(
            api_base=custom_base,
            api_key=None,
            model="text-embedding-3-small",
            optional_params={},
            litellm_params={}
        )
        
        assert url == custom_base
    
    def test_validate_environment_success(self, embedding_config, mock_authenticator):
        """Test successful environment validation."""
        headers = embedding_config.validate_environment(
            headers={},
            model="text-embedding-3-small",
            messages=[],
            optional_params={},
            litellm_params={}
        )
        
        # Check required headers are present
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"
        assert headers["Copilot-Integration-Id"] == "vscode-chat"
        assert headers["Editor-Version"] == "vscode/1.85.0"
        assert "X-Request-Id" in headers
    
    def test_validate_environment_auth_failure(self, embedding_config):
        """Test environment validation with authentication failure."""
        with patch.object(embedding_config.authenticator, 'get_api_key', 
                         side_effect=GetAPIKeyError(status_code=401, message="Auth failed")):
            with pytest.raises(AuthenticationError) as exc_info:
                embedding_config.validate_environment(
                    headers={},
                    model="text-embedding-3-small",
                    messages=[],
                    optional_params={},
                    litellm_params={}
                )
            
            assert "GitHub Copilot authentication failed" in str(exc_info.value)
    
    def test_validate_environment_merge_headers(self, embedding_config, mock_authenticator):
        """Test that custom headers are merged with default headers."""
        custom_headers = {
            "Custom-Header": "custom-value",
            "Authorization": "Bearer custom-token"  # Should be overridden
        }
        
        headers = embedding_config.validate_environment(
            headers=custom_headers,
            model="text-embedding-3-small",
            messages=[],
            optional_params={},
            litellm_params={}
        )
        
        # Custom header should be preserved
        assert headers["Custom-Header"] == "custom-value"
        # But Authorization should be from authenticator
        assert headers["Authorization"] == "Bearer test-api-key"
    
    def test_map_openai_params(self, embedding_config):
        """Test OpenAI parameter mapping."""
        non_default_params = {
            "encoding_format": "float",
            "dimensions": 512,
            "user": "test-user",
            "unsupported_param": "should_be_ignored"
        }
        optional_params = {}
        
        result = embedding_config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="text-embedding-3-small",
            drop_params=True
        )
        
        # Supported params should be mapped
        assert result["encoding_format"] == "float"
        assert result["dimensions"] == 512
        assert result["user"] == "test-user"
        
        # Unsupported params should not be included
        assert "unsupported_param" not in result
    
    def test_transform_embedding_request_single_string(self, embedding_config):
        """Test request transformation with single string input."""
        request = embedding_config.transform_embedding_request(
            model="text-embedding-3-small",
            input="Hello, world!",
            optional_params={"dimensions": 1536},
            headers={}
        )
        
        assert request["input"] == ["Hello, world!"]
        assert request["model"] == "text-embedding-3-small"
        assert request["dimensions"] == 1536
    
    def test_transform_embedding_request_list_input(self, embedding_config):
        """Test request transformation with list input."""
        input_texts = ["Hello", "World", "Test"]
        request = embedding_config.transform_embedding_request(
            model="text-embedding-3-small",
            input=input_texts,
            optional_params={"encoding_format": "float"},
            headers={}
        )
        
        assert request["input"] == input_texts
        assert request["model"] == "text-embedding-3-small"
        assert request["encoding_format"] == "float"
    
    def test_transform_embedding_request_other_input_types(self, embedding_config):
        """Test request transformation with non-string/list input."""
        request = embedding_config.transform_embedding_request(
            model="text-embedding-3-small",
            input=12345,  # Non-string input
            optional_params={},
            headers={}
        )
        
        assert request["input"] == ["12345"]  # Should be converted to string
        assert request["model"] == "text-embedding-3-small"
    
    def test_transform_embedding_request_all_optional_params(self, embedding_config):
        """Test request transformation with all supported optional parameters."""
        request = embedding_config.transform_embedding_request(
            model="text-embedding-3-small",
            input="Test",
            optional_params={
                "encoding_format": "base64",
                "dimensions": 512,
                "user": "test-user",
                "unsupported": "ignored"
            },
            headers={}
        )
        
        assert request["encoding_format"] == "base64"
        assert request["dimensions"] == 512
        assert request["user"] == "test-user"
        assert "unsupported" not in request
    
    def test_transform_embedding_response_success(self, embedding_config):
        """Test successful response transformation."""
        # Mock response data
        response_data = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        
        # Mock HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        
        # Mock logging object
        mock_logging = MagicMock()
        
        # Create model response object
        model_response = EmbeddingResponse()
        
        # Transform response
        result = embedding_config.transform_embedding_response(
            model="text-embedding-3-small",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            api_key="test-key",
            request_data={},
            optional_params={},
            litellm_params={}
        )
        
        # Verify transformed response
        assert result.model == "text-embedding-3-small"
        assert result.data == response_data["data"]
        assert result.object == "list"
        assert isinstance(result.usage, Usage)
        assert result.usage.prompt_tokens == 5
        assert result.usage.total_tokens == 5
    
    def test_transform_embedding_response_missing_model(self, embedding_config):
        """Test response transformation when model is missing from response."""
        response_data = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "usage": {"prompt_tokens": 3, "total_tokens": 3}
        }
        
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        
        model_response = EmbeddingResponse()
        
        result = embedding_config.transform_embedding_response(
            model="text-embedding-3-small",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            api_key="test-key",
            request_data={},
            optional_params={},
            litellm_params={}
        )
        
        # Should use the model parameter as fallback
        assert result.model == "text-embedding-3-small"
    
    def test_transform_embedding_response_missing_usage(self, embedding_config):
        """Test response transformation when usage is missing."""
        response_data = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "model": "text-embedding-3-small"
        }
        
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        
        model_response = EmbeddingResponse()
        
        result = embedding_config.transform_embedding_response(
            model="text-embedding-3-small",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            api_key="test-key",
            request_data={},
            optional_params={},
            litellm_params={}
        )
        
        # Should have default usage values
        assert result.usage.prompt_tokens == 0
        assert result.usage.total_tokens == 0
    
    def test_transform_embedding_response_json_parse_error(self, embedding_config):
        """Test response transformation with JSON parsing error."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.status_code = 200
        mock_response.text = "Invalid JSON response"
        mock_response.headers = {}
        
        model_response = EmbeddingResponse()
        
        with pytest.raises(GithubCopilotError) as exc_info:
            embedding_config.transform_embedding_response(
                model="text-embedding-3-small",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=MagicMock(),
                api_key="test-key",
                request_data={},
                optional_params={},
                litellm_params={}
            )
        
        assert "Failed to parse GitHub Copilot embedding response" in str(exc_info.value)
    
    def test_transform_embedding_response_missing_data_field(self, embedding_config):
        """Test response transformation when data field is missing."""
        response_data = {
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
            # Missing 'data' field
        }
        
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}
        
        model_response = EmbeddingResponse()
        
        with pytest.raises(GithubCopilotError) as exc_info:
            embedding_config.transform_embedding_response(
                model="text-embedding-3-small",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=MagicMock(),
                api_key="test-key",
                request_data={},
                optional_params={},
                litellm_params={}
            )
        
        assert "Invalid response format: missing 'data' field" in str(exc_info.value)
    
    def test_get_error_class(self, embedding_config):
        """Test error class generation."""
        error = embedding_config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"}
        )
        
        assert isinstance(error, GithubCopilotError)
        assert error.status_code == 400
        assert "Test error" in str(error)


class TestGithubCopilotEmbeddingIntegration:
    """Integration tests for GitHub Copilot embedding functionality."""
    
    @pytest.mark.asyncio
    async def test_embedding_e2e_mock(self):
        """Test end-to-end embedding flow with mocked responses."""
        import litellm
        
        # Mock the HTTP handler
        with patch('litellm.llms.custom_httpx.llm_http_handler.embedding') as mock_embedding:
            # Mock response
            mock_response = EmbeddingResponse()
            mock_response.model = "text-embedding-3-small"
            mock_response.data = [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * 1536
                }
            ]
            mock_response.usage = Usage(prompt_tokens=5, total_tokens=5)
            mock_response.object = "list"
            
            mock_embedding.return_value = mock_response
            
            # Test the actual litellm call
            response = await litellm.aembedding(
                model="github_copilot/text-embedding-3-small",
                input="Hello, world!"
            )
            
            # Verify the response
            assert response.model == "text-embedding-3-small"
            assert len(response.data) == 1
            assert len(response.data[0]["embedding"]) == 1536
            assert response.usage.prompt_tokens == 5
            
            # Verify the mock was called correctly
            mock_embedding.assert_called_once()
            call_args = mock_embedding.call_args
            assert call_args[1]["model"] == "text-embedding-3-small"
            assert call_args[1]["custom_llm_provider"] == "github_copilot"
    
    def test_embedding_sync_mock(self):
        """Test synchronous embedding with mocked responses."""
        import litellm
        
        with patch('litellm.llms.custom_httpx.llm_http_handler.embedding') as mock_embedding:
            # Mock response
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
            
            # Test synchronous call
            response = litellm.embedding(
                model="github_copilot/text-embedding-3-small",
                input="Test"
            )
            
            # Verify response
            assert response.model == "text-embedding-3-small"
            assert len(response.data) == 1
            assert response.usage.prompt_tokens == 3
    
    def test_embedding_multiple_inputs_mock(self):
        """Test embedding with multiple input texts."""
        import litellm
        
        with patch('litellm.llms.custom_httpx.llm_http_handler.embedding') as mock_embedding:
            # Mock response for multiple inputs
            mock_response = EmbeddingResponse()
            mock_response.model = "text-embedding-3-small"
            mock_response.data = [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * 1536
                },
                {
                    "object": "embedding",
                    "index": 1,
                    "embedding": [0.2] * 1536
                },
                {
                    "object": "embedding",
                    "index": 2,
                    "embedding": [0.3] * 1536
                }
            ]
            mock_response.usage = Usage(prompt_tokens=15, total_tokens=15)
            
            mock_embedding.return_value = mock_response
            
            # Test with multiple inputs
            response = litellm.embedding(
                model="github_copilot/text-embedding-3-small",
                input=["Hello", "World", "Test"]
            )
            
            # Verify response
            assert response.model == "text-embedding-3-small"
            assert len(response.data) == 3
            assert response.usage.prompt_tokens == 15
    
    def test_embedding_with_optional_params_mock(self):
        """Test embedding with optional parameters."""
        import litellm
        
        with patch('litellm.llms.custom_httpx.llm_http_handler.embedding') as mock_embedding:
            # Mock response with custom dimensions
            mock_response = EmbeddingResponse()
            mock_response.model = "text-embedding-3-small"
            mock_response.data = [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * 512  # Custom dimensions
                }
            ]
            mock_response.usage = Usage(prompt_tokens=4, total_tokens=4)
            
            mock_embedding.return_value = mock_response
            
            # Test with optional parameters
            response = litellm.embedding(
                model="github_copilot/text-embedding-3-small",
                input="Test with custom dimensions",
                dimensions=512,
                user="test-user"
            )
            
            # Verify response
            assert response.model == "text-embedding-3-small"
            assert len(response.data[0]["embedding"]) == 512
            assert response.usage.prompt_tokens == 4
            
            # Verify optional params were passed
            call_args = mock_embedding.call_args
            assert "dimensions" in call_args[1]["optional_params"]
            assert call_args[1]["optional_params"]["dimensions"] == 512
    
    def test_provider_config_manager_integration(self):
        """Test that ProviderConfigManager correctly returns GitHub Copilot embedding config."""
        from litellm.utils import ProviderConfigManager
        from litellm.types.utils import LlmProviders
        
        config = ProviderConfigManager.get_provider_embedding_config(
            model="text-embedding-3-small",
            provider=LlmProviders.GITHUB_COPILOT
        )
        
        assert config is not None
        assert isinstance(config, GithubCopilotEmbeddingConfig)
    
    def test_embedding_error_handling_mock(self):
        """Test error handling in embedding calls."""
        import litellm
        from litellm.exceptions import BadRequestError
        
        with patch('litellm.llms.custom_httpx.llm_http_handler.embedding') as mock_embedding:
            # Mock an error response
            mock_embedding.side_effect = BadRequestError(
                message="Invalid model",
                model="github_copilot/invalid-model",
                llm_provider="github_copilot"
            )
            
            # Test error handling
            with pytest.raises(BadRequestError) as exc_info:
                litellm.embedding(
                    model="github_copilot/invalid-model",
                    input="Test error handling"
                )
            
            assert "Invalid model" in str(exc_info.value)


# Pytest fixtures for shared test data
@pytest.fixture
def sample_embedding_response():
    """Sample embedding response data for testing."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 307 + [0.1, 0.2]  # 1537 values (truncate to 1536)
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }


@pytest.fixture
def sample_multi_embedding_response():
    """Sample multi-text embedding response for testing."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1] * 1536
            },
            {
                "object": "embedding", 
                "index": 1,
                "embedding": [0.2] * 1536
            },
            {
                "object": "embedding",
                "index": 2,
                "embedding": [0.3] * 1536
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 15,
            "total_tokens": 15
        }
    }