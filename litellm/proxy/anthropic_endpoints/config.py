"""
Configuration integration for enhanced Anthropic endpoints.
Handles setup and integration with LiteLLM proxy configuration system.
"""

import os
from typing import Any, Dict, Optional

from litellm._logging import verbose_proxy_logger
from .model_mapping import anthropic_model_mapper, ProviderModelConfig


def initialize_anthropic_endpoints(proxy_config: Optional[Any] = None, general_settings: Optional[Dict] = None):
    """
    Initialize enhanced Anthropic endpoints with configuration.
    
    This function should be called during proxy startup to configure
    model mappings and other Anthropic-specific settings.
    
    Args:
        proxy_config: The LiteLLM proxy configuration object
        general_settings: General settings dictionary from proxy config
    """
    try:
        verbose_proxy_logger.info("🚀 INIT: Initializing enhanced Anthropic endpoints...")
        verbose_proxy_logger.info(f"🔍 INIT DEBUG: proxy_config provided: {proxy_config is not None}")
        verbose_proxy_logger.info(f"🔍 INIT DEBUG: general_settings provided: {general_settings is not None}")
        
        if general_settings is not None:
            verbose_proxy_logger.info(f"🔍 INIT DEBUG: general_settings keys: {list(general_settings.keys()) if isinstance(general_settings, dict) else 'Not a dict'}")
        
        # Configure from proxy config if available
        if proxy_config is not None:
            verbose_proxy_logger.info("🔧 INIT: Configuring from proxy_config...")
            configure_from_proxy_config(proxy_config)
        
        # Configure from general settings
        if general_settings is not None:
            verbose_proxy_logger.info("🔧 INIT: Configuring from general_settings...")
            configure_from_general_settings(general_settings)
        
        # Configure from environment variables as fallback
        verbose_proxy_logger.info("🔧 INIT: Configuring from environment variables...")
        configure_from_environment()
        
        # Direct fallback: Try to load from known config files
        verbose_proxy_logger.info("🔧 INIT: Attempting direct config file fallback...")
        configure_from_config_files()
        
        # Log final state
        verbose_proxy_logger.info(f"🔍 INIT DEBUG: Final default provider: {anthropic_model_mapper.default_provider}")
        verbose_proxy_logger.info(f"🔍 INIT DEBUG: Available providers: {list(anthropic_model_mapper.provider_configs.keys())}")
        
        verbose_proxy_logger.info("✅ INIT: Enhanced Anthropic endpoints initialized successfully")
        
    except Exception as e:
        verbose_proxy_logger.warning(f"❌ INIT ERROR: Failed to initialize enhanced Anthropic endpoints: {e}")
        import traceback
        verbose_proxy_logger.debug(f"INIT ERROR TRACEBACK: {traceback.format_exc()}")


def configure_from_proxy_config(proxy_config: Any):
    """Configure model mappings from proxy configuration object."""
    try:
        # Check if there's Anthropic configuration in the proxy config
        if hasattr(proxy_config, 'general_settings'):
            general_settings = proxy_config.general_settings
            if 'anthropic' in general_settings:
                anthropic_config = general_settings['anthropic']
                _apply_anthropic_config(anthropic_config)
                
        # Check router configuration for model mappings
        if hasattr(proxy_config, 'router_settings') and proxy_config.router_settings:
            router_settings = proxy_config.router_settings
            if 'anthropic_model_mapping' in router_settings:
                _apply_model_mapping_config(router_settings['anthropic_model_mapping'])
                
    except Exception as e:
        verbose_proxy_logger.debug(f"Could not configure from proxy config: {e}")


def configure_from_general_settings(general_settings: Dict):
    """Configure model mappings from general settings dictionary."""
    try:
        verbose_proxy_logger.info(f"🔍 CONFIG DEBUG: configure_from_general_settings called")
        verbose_proxy_logger.info(f"🔍 CONFIG DEBUG: general_settings keys: {list(general_settings.keys()) if general_settings else 'None'}")
        verbose_proxy_logger.info(f"🔍 CONFIG DEBUG: general_settings type: {type(general_settings)}")
        
        if 'anthropic' in general_settings:
            anthropic_config = general_settings['anthropic']
            verbose_proxy_logger.info(f"🔍 CONFIG DEBUG: Found 'anthropic' in general_settings: {anthropic_config}")
            _apply_anthropic_config(anthropic_config)
        else:
            verbose_proxy_logger.warning("❌ CONFIG: No 'anthropic' key found in general_settings")
            verbose_proxy_logger.info(f"🔍 CONFIG DEBUG: Available keys in general_settings: {list(general_settings.keys()) if isinstance(general_settings, dict) else 'Not a dict'}")
            
        if 'anthropic_model_mapping' in general_settings:
            verbose_proxy_logger.info(f"🔍 CONFIG DEBUG: Found 'anthropic_model_mapping' in general_settings")
            _apply_model_mapping_config(general_settings['anthropic_model_mapping'])
        else:
            verbose_proxy_logger.info("🔍 CONFIG DEBUG: No 'anthropic_model_mapping' in general_settings")
            
    except Exception as e:
        verbose_proxy_logger.error(f"❌ CONFIG ERROR: Could not configure from general settings: {e}")
        import traceback
        verbose_proxy_logger.debug(f"CONFIG ERROR TRACEBACK: {traceback.format_exc()}")


def configure_from_environment():
    """Configure model mappings from environment variables."""
    try:
        # Configure default provider based on available API keys
        default_provider = None
        
        # Check for various provider API keys
        if os.getenv("OPENAI_API_KEY"):
            default_provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            default_provider = "anthropic"
        elif os.getenv("GITHUB_TOKEN"):
            default_provider = "github_copilot"
        elif os.getenv("MODELSCOPE_API_KEY"):
            default_provider = "modelscope"
        
        if default_provider:
            verbose_proxy_logger.debug(f"Setting default provider to {default_provider} based on available API keys")
            _configure_default_provider(default_provider)
            
        # Check for custom model mapping environment variable
        custom_mapping = os.getenv("ANTHROPIC_MODEL_MAPPING")
        if custom_mapping:
            try:
                import json
                mapping_config = json.loads(custom_mapping)
                _apply_model_mapping_config(mapping_config)
                verbose_proxy_logger.debug("Applied custom model mapping from environment")
            except json.JSONDecodeError as e:
                verbose_proxy_logger.warning(f"Invalid JSON in ANTHROPIC_MODEL_MAPPING: {e}")
                
    except Exception as e:
        verbose_proxy_logger.debug(f"Could not configure from environment: {e}")


def configure_from_config_files():
    """Direct fallback: Try to load configuration from known config files."""
    try:
        import yaml
        
        # List of potential config file paths
        config_paths = [
            "anthropic_iflow_qwen3_coder_config.yaml",
            "config.yaml",
            "proxy_server_config.yaml",
            os.getenv("CONFIG_FILE_PATH", ""),
            os.getenv("WORKER_CONFIG", "")
        ]
        
        for config_path in config_paths:
            if not config_path or not os.path.isfile(config_path):
                continue
                
            verbose_proxy_logger.info(f"🔍 FALLBACK: Trying to load config from: {config_path}")
            
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                if not config:
                    continue
                    
                general_settings = config.get('general_settings', {})
                if 'anthropic' in general_settings:
                    verbose_proxy_logger.info(f"✅ FALLBACK: Found anthropic config in {config_path}")
                    configure_from_general_settings(general_settings)
                    return  # Exit after first successful configuration
                    
            except Exception as e:
                verbose_proxy_logger.debug(f"Could not load config from {config_path}: {e}")
                continue
        
        verbose_proxy_logger.info("🔍 FALLBACK: No valid anthropic config found in any config file")
        
    except Exception as e:
        verbose_proxy_logger.debug(f"Could not configure from config files: {e}")


def _apply_anthropic_config(anthropic_config: Dict):
    """Apply Anthropic-specific configuration."""
    try:
        verbose_proxy_logger.info(f"🔍 CONFIG DEBUG: _apply_anthropic_config called with: {anthropic_config}")
        
        # Clear existing provider configs to avoid conflicts with defaults
        verbose_proxy_logger.info("🔄 CONFIG: Resetting Anthropic model mapper for new configuration")
        anthropic_model_mapper.provider_configs.clear()
        anthropic_model_mapper.default_provider = None
        
        # Update model mappings if provided
        if 'model_mappings' in anthropic_config:
            mappings = anthropic_config['model_mappings']
            verbose_proxy_logger.info(f"🔍 CONFIG DEBUG: Found model_mappings: {mappings}")
            
            # Update Anthropic model tier definitions
            if 'anthropic_models' in mappings:
                anthropic_model_mapper.update_anthropic_models(mappings['anthropic_models'])
                verbose_proxy_logger.info(f"✅ CONFIG: Updated Anthropic model tier mappings: {mappings['anthropic_models']}")
                
            # Configure provider mappings
            if 'providers' in mappings:
                verbose_proxy_logger.info(f"🔍 CONFIG DEBUG: Found providers: {list(mappings['providers'].keys())}")
                for provider_name, provider_data in mappings['providers'].items():
                    verbose_proxy_logger.info(f"🔍 CONFIG DEBUG: Configuring provider {provider_name}: {provider_data}")
                    config = ProviderModelConfig(
                        name=provider_name,
                        prefix=provider_data.get('prefix'),
                        models=provider_data.get('models', {}),
                        extra_headers=provider_data.get('extra_headers', {})
                    )
                    is_default = provider_data.get('default', False)
                    anthropic_model_mapper.register_provider(config, is_default)
                    verbose_proxy_logger.info(f"✅ CONFIG: Registered provider {provider_name} (default: {is_default})")
                    verbose_proxy_logger.info(f"🔍 CONFIG DEBUG: Provider {provider_name} models: {config.models}")
        else:
            verbose_proxy_logger.warning("❌ CONFIG: No 'model_mappings' found in anthropic_config")
                    
        verbose_proxy_logger.info("✅ CONFIG: Applied Anthropic configuration successfully")
        
        # Log final state
        verbose_proxy_logger.info(f"🔍 CONFIG DEBUG: Final anthropic_model_mapper state:")
        verbose_proxy_logger.info(f"   - Available providers: {list(anthropic_model_mapper.provider_configs.keys())}")
        verbose_proxy_logger.info(f"   - Default provider: {anthropic_model_mapper.default_provider}")
        verbose_proxy_logger.info(f"   - Anthropic models: {anthropic_model_mapper.anthropic_models}")
        
    except Exception as e:
        verbose_proxy_logger.error(f"❌ CONFIG ERROR: Failed to apply Anthropic config: {e}")
        import traceback
        verbose_proxy_logger.debug(f"CONFIG ERROR TRACEBACK: {traceback.format_exc()}")


def _apply_model_mapping_config(mapping_config: Dict):
    """Apply model mapping configuration."""
    try:
        # Handle direct provider configurations
        for provider_name, provider_data in mapping_config.items():
            if isinstance(provider_data, dict):
                config = ProviderModelConfig(
                    name=provider_name,
                    prefix=provider_data.get('prefix'),
                    models=provider_data.get('models', {}),
                    extra_headers=provider_data.get('extra_headers', {})
                )
                is_default = provider_data.get('default', False)
                anthropic_model_mapper.register_provider(config, is_default)
                
        verbose_proxy_logger.debug("Applied model mapping configuration")
        
    except Exception as e:
        verbose_proxy_logger.warning(f"Failed to apply model mapping config: {e}")


def _configure_default_provider(provider_name: str):
    """Configure a default provider with sensible defaults."""
    try:
        if provider_name == "openai":
            config = ProviderModelConfig(
                name="openai",
                prefix="openai/",
                models={
                    "big": "gpt-4o",
                    "medium": "gpt-4o-mini",
                    "small": "gpt-3.5-turbo"
                }
            )
            anthropic_model_mapper.register_provider(config, is_default=True)
            
        elif provider_name == "anthropic":
            config = ProviderModelConfig(
                name="anthropic", 
                prefix="anthropic/",
                models={
                    "big": "claude-3-opus-20240229",
                    "medium": "claude-3-5-sonnet-20241022",
                    "small": "claude-3-haiku-20240307"
                }
            )
            anthropic_model_mapper.register_provider(config, is_default=True)
            
        elif provider_name == "github_copilot":
            config = ProviderModelConfig(
                name="github_copilot",
                prefix="github_copilot/",
                models={
                    "big": "gpt-4o",
                    "medium": "gpt-4o",
                    "small": "gpt-4o-mini"
                },
                extra_headers={
                    "Editor-Version": "vscode/1.85.0",
                    "Editor-Plugin-Version": "copilot-chat/0.11.1",
                    "User-Agent": "GitHubCopilot/1.0",
                    "Copilot-Integration-Id": "vscode-chat",
                }
            )
            anthropic_model_mapper.register_provider(config, is_default=True)
            
        elif provider_name == "modelscope":
            config = ProviderModelConfig(
                name="modelscope",
                prefix="dashscope/",
                models={
                    "big": "Qwen/Qwen3-235B-A22B-Instruct-2507",
                    "medium": "Qwen/Qwen3-235B-A22B-Instruct-2507",
                    "small": "Qwen/Qwen3-235B-A22B-Instruct-2507"
                }
            )
            anthropic_model_mapper.register_provider(config, is_default=True)
            
        verbose_proxy_logger.debug(f"Configured default provider: {provider_name}")
        
    except Exception as e:
        verbose_proxy_logger.warning(f"Failed to configure default provider {provider_name}: {e}")


# Example configuration that can be added to proxy config YAML:
EXAMPLE_ANTHROPIC_CONFIG = """
# Add this to your LiteLLM proxy configuration YAML:

general_settings:
  anthropic:
    model_mappings:
      # Define which tier each Anthropic model belongs to
      anthropic_models:
        claude-3-opus-20240229: big
        claude-3-sonnet-20240229: medium
        claude-3-haiku-20240307: small
        claude-3-5-sonnet-20241022: medium
        claude-3-5-haiku-20241022: small
        
      # Define provider-specific model mappings
      providers:
        openai:
          prefix: "openai/"
          default: true
          models:
            big: "gpt-4o"
            medium: "gpt-4o-mini"
            small: "gpt-3.5-turbo"
            
        github_copilot:
          prefix: "github_copilot/"
          models:
            big: "gpt-4o"
            medium: "gpt-4o"
            small: "gpt-4o-mini"
          extra_headers:
            Editor-Version: "vscode/1.85.0"
            Editor-Plugin-Version: "copilot-chat/0.11.1"
            User-Agent: "GitHubCopilot/1.0"
            Copilot-Integration-Id: "vscode-chat"
            
        modelscope:
          prefix: "dashscope/"
          models:
            big: "Qwen/Qwen3-235B-A22B-Instruct-2507"
            medium: "Qwen/Qwen3-235B-A22B-Instruct-2507"
            small: "Qwen/Qwen3-235B-A22B-Instruct-2507"
"""