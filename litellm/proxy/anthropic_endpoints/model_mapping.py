"""
Model tier mapping system for Anthropic API compatibility.
Maps Anthropic models to provider-specific models via tiers (big/medium/small).
"""

import os
from typing import Dict, Optional, Any
from dataclasses import dataclass

import litellm
from litellm._logging import verbose_proxy_logger


@dataclass
class ProviderModelConfig:
    """Configuration for a provider's model mappings."""
    
    name: str
    prefix: Optional[str] = None
    models: Dict[str, str] = None  # tier -> model mapping
    extra_headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = {}
        if self.extra_headers is None:
            self.extra_headers = {}


class AnthropicModelMapper:
    """
    Maps Anthropic models to provider-specific models using a tier-based system.
    
    This allows clients to use standard Anthropic model names (e.g., claude-3-sonnet-20240229)
    while the proxy routes to appropriate provider models based on configured tiers.
    """
    
    # Default Anthropic model tier classifications
    DEFAULT_ANTHROPIC_MODELS = {
        "claude-3-opus-20240229": "big",
        "claude-3-sonnet-20240229": "medium", 
        "claude-3-haiku-20240307": "small",
        "claude-3-5-sonnet-20240620": "medium",
        "claude-3-5-sonnet-20241022": "medium",
        "claude-3-5-sonnet-latest": "medium",
        "claude-3-5-haiku-20241022": "small",
        "claude-3-5-haiku-latest": "small",
        "claude-sonnet-4-20250514": "big",
        "claude-sonnet-4-5-20250929": "big",
        "claude-3-7-sonnet-20250219": "big",
        "claude-opus-4-1-20250805": "big",
    }
    
    def __init__(self):
        self.anthropic_models = self.DEFAULT_ANTHROPIC_MODELS.copy()
        self.provider_configs: Dict[str, ProviderModelConfig] = {}
        self.default_provider = None
        
    def register_provider(self, config: ProviderModelConfig, is_default: bool = False):
        """Register a provider with its model mappings."""
        self.provider_configs[config.name] = config
        if is_default:
            self.default_provider = config.name
            
        verbose_proxy_logger.debug(
            f"Registered provider {config.name} with models: {config.models}"
        )
    
    def update_anthropic_models(self, models: Dict[str, str]):
        """Update the Anthropic model to tier mappings."""
        self.anthropic_models.update(models)
        verbose_proxy_logger.debug(f"Updated Anthropic model mappings: {models}")
    
    def get_anthropic_model_tier(self, anthropic_model: str) -> str:
        """Get the tier (big/medium/small) for an Anthropic model."""
        tier = self.anthropic_models.get(anthropic_model)
        if not tier:
            # Default to medium if model not found
            verbose_proxy_logger.warning(
                f"Unknown Anthropic model '{anthropic_model}', defaulting to 'medium' tier"
            )
            return "medium"
        return tier
    
    def resolve_model(self, anthropic_model: str, provider_name: Optional[str] = None) -> str:
        """
        Resolve an Anthropic model name to a provider-specific model.
        
        Args:
            anthropic_model: The Anthropic model name (e.g., "claude-3-sonnet-20240229")
            provider_name: Optional specific provider to use
            
        Returns:
            The provider-specific model name
            
        Raises:
            ValueError: If no appropriate model mapping is found
        """
        verbose_proxy_logger.info(f"🔍 RESOLVE_MODEL START: anthropic_model={anthropic_model}, provider_name={provider_name}")
        
        # Determine which provider to use
        target_provider = provider_name or self.default_provider
        verbose_proxy_logger.info(f"🔍 RESOLVE_MODEL: target_provider={target_provider}")
        
        if not target_provider:
            error_msg = "No provider specified and no default provider configured"
            verbose_proxy_logger.error(f"❌ RESOLVE_MODEL ERROR: {error_msg}")
            raise ValueError(error_msg)
            
        provider_config = self.provider_configs.get(target_provider)
        verbose_proxy_logger.info(f"🔍 RESOLVE_MODEL: provider_config found={provider_config is not None}")
        
        if not provider_config:
            error_msg = f"Provider '{target_provider}' not registered"
            verbose_proxy_logger.error(f"❌ RESOLVE_MODEL ERROR: {error_msg}")
            verbose_proxy_logger.info(f"🔍 RESOLVE_MODEL: Available providers: {list(self.provider_configs.keys())}")
            raise ValueError(error_msg)
        
        # Get the tier for this Anthropic model
        tier = self.get_anthropic_model_tier(anthropic_model)
        verbose_proxy_logger.info(f"🔍 RESOLVE_MODEL: tier={tier}")
        
        # Get the provider-specific model for this tier
        provider_model = provider_config.models.get(tier)
        verbose_proxy_logger.info(f"🔍 RESOLVE_MODEL: provider_model={provider_model}")
        verbose_proxy_logger.info(f"🔍 RESOLVE_MODEL: provider_config.models={provider_config.models}")
        
        if not provider_model:
            error_msg = f"No model defined for tier '{tier}' in provider '{target_provider}'"
            verbose_proxy_logger.error(f"❌ RESOLVE_MODEL ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Apply prefix if defined
        verbose_proxy_logger.info(f"🔍 RESOLVE_MODEL: provider_config.prefix={provider_config.prefix}")
        if provider_config.prefix:
            resolved_model = f"{provider_config.prefix}{provider_model}"
        else:
            resolved_model = provider_model
            
        verbose_proxy_logger.info(
            f"🔍 RESOLVE_MODEL DEBUG: {anthropic_model} -> tier:{tier} -> {resolved_model} (provider:{target_provider})"
        )
        
        return resolved_model
    
    def get_provider_extra_headers(self, provider_name: Optional[str] = None) -> Dict[str, str]:
        """Get extra headers for a provider (e.g., for GitHub Copilot)."""
        target_provider = provider_name or self.default_provider
        if not target_provider:
            return {}
            
        provider_config = self.provider_configs.get(target_provider)
        if not provider_config:
            return {}
            
        return provider_config.extra_headers.copy()
    
    def list_available_models(self) -> list:
        """
        List all available Anthropic models that can be mapped.
        
        Returns a list of model objects in Anthropic API format.
        """
        models = []
        for model_name in self.anthropic_models.keys():
            models.append({
                "id": model_name,
                "object": "model",
                "created": 1708726751,  # Default timestamp
                "owned_by": "anthropic",
            })
        
        # Add some standard models that should always be available
        standard_models = [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229", 
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ]
        
        for model_name in standard_models:
            if not any(m["id"] == model_name for m in models):
                models.append({
                    "id": model_name,
                    "object": "model", 
                    "created": 1708726751,
                    "owned_by": "anthropic",
                })
        
        return models


# Global instance
anthropic_model_mapper = AnthropicModelMapper()


def configure_from_proxy_config(proxy_config: Any):
    """
    Configure the model mapper from LiteLLM proxy configuration.
    
    This function looks for Anthropic-specific configuration in the proxy config
    and sets up model mappings accordingly.
    """
    global anthropic_model_mapper
    
    try:
        # Check if there's Anthropic configuration in the proxy config
        general_settings = getattr(proxy_config, 'general_settings', {})
        anthropic_config = general_settings.get('anthropic', {})
        
        if not anthropic_config:
            verbose_proxy_logger.debug("No Anthropic configuration found in proxy config")
            return
            
        # Clear existing provider configs to avoid conflicts with defaults
        verbose_proxy_logger.info("🔄 CONFIG: Resetting Anthropic model mapper for new configuration")
        anthropic_model_mapper.provider_configs.clear()
        anthropic_model_mapper.default_provider = None
            
        # Update model mappings if provided
        if 'model_mappings' in anthropic_config:
            mappings = anthropic_config['model_mappings']
            if 'anthropic_models' in mappings:
                anthropic_model_mapper.update_anthropic_models(mappings['anthropic_models'])
                
            # Configure provider mappings
            if 'providers' in mappings:
                for provider_name, provider_data in mappings['providers'].items():
                    config = ProviderModelConfig(
                        name=provider_name,
                        prefix=provider_data.get('prefix'),
                        models=provider_data.get('models', {}),
                        extra_headers=provider_data.get('extra_headers', {})
                    )
                    is_default = provider_data.get('default', False)
                    anthropic_model_mapper.register_provider(config, is_default)
                    
        verbose_proxy_logger.info("Configured Anthropic model mapper from proxy config")
        
    except Exception as e:
        verbose_proxy_logger.warning(f"Failed to configure Anthropic model mapper: {e}")


def configure_default_mappings():
    """
    Configure default model mappings for common providers.
    
    This is used as a fallback when no specific configuration is provided.
    """
    global anthropic_model_mapper
    
    # Default OpenAI mapping
    openai_config = ProviderModelConfig(
        name="openai",
        prefix="openai/",
        models={
            "big": "gpt-4o",
            "medium": "gpt-4o-mini", 
            "small": "gpt-3.5-turbo"
        }
    )
    anthropic_model_mapper.register_provider(openai_config, is_default=True)
    
    # Default Anthropic mapping (direct)
    anthropic_config = ProviderModelConfig(
        name="anthropic",
        prefix="anthropic/",
        models={
            "big": "claude-3-opus-20240229",
            "medium": "claude-3-5-sonnet-20241022",
            "small": "claude-3-haiku-20240307"
        }
    )
    anthropic_model_mapper.register_provider(anthropic_config)
    
    verbose_proxy_logger.debug("Configured default Anthropic model mappings")


# Configure defaults on import
configure_default_mappings()