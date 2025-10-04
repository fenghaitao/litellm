"""
GitHub Copilot Embedding API transformation.

This module provides transformation logic for GitHub Copilot embedding requests and responses,
integrating with the existing GitHub Copilot authentication system.
"""

from typing import List, Optional, Union, Dict, Any
import httpx
import uuid

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage
from litellm.exceptions import AuthenticationError

from ..authenticator import Authenticator
from ..common_utils import GithubCopilotError, GetAPIKeyError


class GithubCopilotEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration class for GitHub Copilot embedding API.
    
    This class handles request/response transformation for GitHub Copilot's embedding endpoint,
    reusing the existing authentication infrastructure from the chat implementation.
    """
    
    GITHUB_COPILOT_API_BASE = "https://api.githubcopilot.com/"
    
    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()
    
    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for GitHub Copilot embedding requests.
        
        Args:
            api_base: Base API URL (optional, will use default if not provided)
            api_key: API key (handled by authenticator)
            model: Model name
            optional_params: Additional parameters
            litellm_params: LiteLLM parameters
            stream: Streaming flag (not used for embeddings)
            
        Returns:
            Complete URL for the embedding endpoint
        """
        # Use dynamic API base from authenticator or fallback to default
        base_url = (
            self.authenticator.get_api_base() or 
            api_base or 
            self.GITHUB_COPILOT_API_BASE
        )
        
        # Ensure the URL ends with embeddings endpoint
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/embeddings"):
            base_url = f"{base_url}/embeddings"
            
        return base_url
    
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate environment and return headers for GitHub Copilot embedding requests.
        
        Args:
            headers: Input headers
            model: Model name
            messages: Messages (not used for embeddings)
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            api_key: API key (handled by authenticator)
            api_base: API base URL
            
        Returns:
            Dict of validated headers
            
        Raises:
            AuthenticationError: If authentication fails
        """
        try:
            # Get API key from authenticator
            dynamic_api_key = self.authenticator.get_api_key()
        except GetAPIKeyError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="github_copilot",
                message=f"GitHub Copilot authentication failed: {str(e)}",
            )
        
        # Build headers with GitHub Copilot specific requirements
        validated_headers = {
            "Authorization": f"Bearer {dynamic_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Copilot-Integration-Id": "vscode-chat",
            "Editor-Version": "vscode/1.85.0",
            "Editor-Plugin-Version": "copilot-chat/0.26.7",
            "User-Agent": "GitHubCopilotChat/0.26.7",
            "OpenAI-Intent": "conversation-panel",
            "X-GitHub-Api-Version": "2025-04-01",
            "X-Request-Id": str(uuid.uuid4()),
            "X-VSCode-User-Agent-Library-Version": "electron-fetch",
        }
        
        # Merge with any additional headers provided
        validated_headers.update(headers)
        
        return validated_headers
    
    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for GitHub Copilot embeddings.
        
        Args:
            model: Model name
            
        Returns:
            List of supported parameter names
        """
        return [
            "encoding_format",
            "dimensions", 
            "user",
        ]
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI embedding parameters to GitHub Copilot format.
        
        Args:
            non_default_params: Non-default parameters from request
            optional_params: Optional parameters dict to update
            model: Model name
            drop_params: Whether to drop unsupported params
            
        Returns:
            Updated optional_params dict
        """
        supported_params = self.get_supported_openai_params(model)
        
        for param_name in supported_params:
            if param_name in non_default_params:
                optional_params[param_name] = non_default_params[param_name]
        
        return optional_params
    
    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform embedding request to GitHub Copilot format.
        
        Args:
            model: Model name
            input: Input text(s) to embed
            optional_params: Optional parameters
            headers: Request headers
            
        Returns:
            Transformed request payload
        """
        # Ensure input is a list of strings
        if isinstance(input, str):
            input_list = [input]
        elif isinstance(input, list):
            input_list = input
        else:
            # Handle other input types by converting to string
            input_list = [str(input)]
        
        # Build request payload compatible with OpenAI embedding format
        request_payload = {
            "input": input_list,
            "model": model,
        }
        
        # Add supported optional parameters
        for param in ["encoding_format", "dimensions", "user"]:
            if param in optional_params:
                request_payload[param] = optional_params[param]
        
        return request_payload
    
    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> EmbeddingResponse:
        """
        Transform GitHub Copilot embedding response to standard format.
        
        Args:
            model: Model name
            raw_response: Raw HTTP response
            model_response: EmbeddingResponse object to populate
            logging_obj: Logging object
            api_key: API key (optional)
            request_data: Original request data
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            
        Returns:
            Populated EmbeddingResponse object
            
        Raises:
            GithubCopilotError: If response parsing fails
        """
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise GithubCopilotError(
                status_code=raw_response.status_code,
                message=f"Failed to parse GitHub Copilot embedding response: {str(e)}",
                headers=raw_response.headers,
                body={"error": raw_response.text},
            )
        
        # Validate response structure
        if "data" not in response_json:
            raise GithubCopilotError(
                status_code=raw_response.status_code,
                message="Invalid response format: missing 'data' field",
                headers=raw_response.headers,
                body=response_json,
            )
        
        # Populate model response fields
        model_response.model = response_json.get("model", model)
        model_response.data = response_json["data"]
        model_response.object = response_json.get("object", "list")
        
        # Extract and set usage information
        usage_data = response_json.get("usage", {})
        model_response.usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        
        return model_response
    
    def get_error_class(
        self, 
        error_message: str, 
        status_code: int, 
        headers: Union[dict, httpx.Headers]
    ) -> GithubCopilotError:
        """
        Get appropriate error class for GitHub Copilot embedding errors.
        
        Args:
            error_message: Error message
            status_code: HTTP status code
            headers: Response headers
            
        Returns:
            GithubCopilotError instance
        """
        return GithubCopilotError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )