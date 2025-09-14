"""Model configuration management"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for a single model"""
    name: str
    provider: str
    api_key_env: str
    rate_limit: int = 60
    timeout: int = 60
    enabled: bool = True
    additional_params: Optional[Dict] = None

class ModelConfigs:
    """Manages configurations for all LLM models"""
    
    # Default model configurations
    DEFAULT_CONFIGS = {
        # OpenAI Models
        # https://platform.openai.com/docs/models
        "o4-mini": ModelConfig(
            name="o4-mini-2025-04-16",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            rate_limit=50,
            timeout=60
        ),
        "03-mini": ModelConfig(
            name="o3-mini-2025-01-31",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            rate_limit=50,
            timeout=60
        ),
        "o1": ModelConfig(
            name="o1-2024-12-17",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            rate_limit=50,
            timeout=60
        ),
       "gpt_4_1": ModelConfig(
            name="gpt-4.1-2025-04-14",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            rate_limit=50,
            timeout=60
        ),
        "gpt_4": ModelConfig(
            name="gpt-4",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            rate_limit=50,
            timeout=60
        ),
        "gpt_3_5_turbo": ModelConfig(
            name="gpt-3.5-turbo", 
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            rate_limit=60,
            timeout=30
        ),
        
        # Anthropic Models
        # https://docs.anthropic.com/en/docs/about-claude/models/overview
        "claude-4-opus": ModelConfig(
            name="claude-opus-4-20250514",
            provider="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            rate_limit=50,
            timeout=60
        ),
        "claude-4-sonnet": ModelConfig(
            name="claude-sonnet-4-20250514",
            provider="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            rate_limit=50,
            timeout=60
        ),
        "claude-3-5-sonnet": ModelConfig(
            name="claude-3-5-sonnet-20241022",
            provider="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            rate_limit=50,
            timeout=60
        ),
        "claude-3-5-haiku": ModelConfig(
            name="claude-3-5-haiku-20241022",
            provider="anthropic", 
            api_key_env="ANTHROPIC_API_KEY",
            rate_limit=50,
            timeout=30
        ),
        "claude-3-haiku": ModelConfig(
            name="claude-3-haiku-20240307",
            provider="anthropic", 
            api_key_env="ANTHROPIC_API_KEY",
            rate_limit=50,
            timeout=30
        ),

        # Cohere Models
        "command-a-03-2025": ModelConfig(
            name="command-a-03-2025",
            provider="cohere",
            api_key_env="COHERE_API_KEY",
            rate_limit=10,
            timeout=120
        ),
        "command-r-plus-08-2024": ModelConfig(
            name="command-r-plus-08-2024",
            provider="cohere",
            api_key_env="COHERE_API_KEY",
            rate_limit=10,
            timeout=120
        ),
        "command-r-08-2024": ModelConfig(
            name="command-r-08-2024",
            provider="cohere",
            api_key_env="COHERE_API_KEY",
            rate_limit=10,
            timeout=120
        ),
        "command": ModelConfig(
            name="command",
            provider="cohere",
            api_key_env="COHERE_API_KEY",
            rate_limit=10,
            timeout=120
        ),

        # Google Models
        "gemini_2_5_pro": ModelConfig(
            name="gemini-2.5-pro",
            provider="google",
            api_key_env="GOOGLE_API_KEY",
            rate_limit=60,
            timeout=60
        ),
        "gemini_2_5_flash": ModelConfig(
            name="gemini-2.5-flash",
            provider="google",
            api_key_env="GOOGLE_API_KEY",
            rate_limit=60,
            timeout=60
        ),
        "gemini_2_0_flash": ModelConfig(
            name="gemini-2.0-flash",
            provider="google",
            api_key_env="GOOGLE_API_KEY",
            rate_limit=60,
            timeout=60
        ),
        
        # xAI Models 
        "grok_4_0709": ModelConfig(
            name="grok-4-0709",
            provider="xai",
            api_key_env="XAI_API_KEY",
            rate_limit=30,
            timeout=60
        ),
        "grok_3_latest": ModelConfig(
            name="grok-3-latest",
            provider="xai",
            api_key_env="XAI_API_KEY",
            rate_limit=30,
            timeout=60
        ),
        "grok_2_vision": ModelConfig(
            name="grok-2-vision-1212",
            provider="xai",
            api_key_env="XAI_API_KEY",
            rate_limit=30,
            timeout=60
        ),
        # HuggingFace Router Models (OpenAI-compatible API)
        # These models are accessed through HF's intelligent routing system microsoft/phi-4

        # Qwen models from Alibaba
        # https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f
        "Qwen3_235b": ModelConfig(
            name="Qwen/Qwen3-235B-A22B-Instruct-2507:novita",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        "Qwen3_32b": ModelConfig(
            name="Qwen/Qwen3-32B:featherless-ai",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        "Qwen3_14b": ModelConfig(
            name="Qwen/Qwen3-14B:featherless-ai",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        "Qwen3_8b": ModelConfig(
            name="Qwen/Qwen3-8B:featherless-ai",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        # HuggingFace Router Models (OpenAI-compatible API)
        # These models are accessed through HF's intelligent routing system microsoft/phi-4
        "Qwen3_4b": ModelConfig(
            name="Qwen/Qwen3-4B:nebius",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        #llama models:
        "llama4_17b_maverick": ModelConfig(
            name="meta-llama/Llama-4-Maverick-17B-128E-Instruct:fireworks-ai",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        "llama4_17b_scout": ModelConfig(
            name="meta-llama/Llama-4-Scout-17B-16E-Instruct:novita",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        "llama3_3_70b": ModelConfig(
            name="meta-llama/Llama-3.3-70B-Instruct:featherless-ai",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        "llama3_2_3b": ModelConfig(
            name="meta-llama/Llama-3.2-3B-Instruct:featherless-ai",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        "llama3_2_1b": ModelConfig(
            name="meta-llama/Llama-3.2-1B-Instruct:featherless-ai",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        # mistral models
        "mistral_3_1_24b": ModelConfig(
            name="mistralai/Mistral-Small-3.1-24B-Instruct-2503", # Try without provider specification
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"},
            enabled=False  # Disabled: No provider available on HuggingFace
        ),
        "mistral_7b": ModelConfig(
            name="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        # "mistral_12b": ModelConfig(
        #     name="mistralai/Mistral-Nemo-Instruct-2407", # drop-in replacement of mistral_7b
        #     provider="huggingface",
        #     api_key_env="HUGGINGFACE_API_KEY",
        #     rate_limit=30,
        #     timeout=120,  # Increased timeout for local inference
        #     additional_params={"type": "local"}
        # ),
        # DeepSeek models via HF Router (with provider specification)
        # https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d
        "deepseek_r1_685b": ModelConfig(
            name="deepseek-ai/DeepSeek-R1-0528:novita",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        "deepseek_r1_distill_llama_70b": ModelConfig(
            name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B:novita",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        "deepseek_r1_distill_qwen_32b": ModelConfig(
            name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:novita",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        "deepseek_r1_distill_qwen_14b": ModelConfig(
            name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B:novita",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        "deepseek_r1_distill_llama_8b": ModelConfig(
            name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B:novita",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),
        # moonshot AI models
        "kimi-k2": ModelConfig(
            name="moonshotai/Kimi-K2-Instruct:novita",
            provider="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            rate_limit=30,
            timeout=90,
            additional_params={"type": "hf_router_api"}
        ),

        # Huggingface local models 
        # ========================
        # "microsoft/phi-2": ModelConfig(
        #     name="microsoft/phi-2",
        #     provider="huggingface",
        #     api_key_env="HUGGINGFACE_API_KEY",
        #     rate_limit=30,
        #     timeout=90,
        #     additional_params={"type": "local"}
        # ),
        # "meta-llama/CodeLlama-34b-Instruct-hf": ModelConfig(
        #     name="meta-llama/CodeLlama-34b-Instruct-hf",
        #     provider="meta",
        #     api_key_env="TOGETHER_API_KEY", 
        #     rate_limit=40,
        #     timeout=60,
        #     additional_params={"provider": "together"}
        # ),
        # "meta-llama/CodeLlama-7b-Instruct-hf": ModelConfig(
        #     name="meta-llama/CodeLlama-7b-Instruct-hf",
        #     provider="huggingface",
        #     api_key_env="HUGGINGFACE_API_KEY",
        #     rate_limit=30,
        #     timeout=90,
        #     additional_params={"type": "local"}
        # ),
        # "bigcode/starcoder2-7b": ModelConfig(
        #     name="bigcode/starcoder2-7b",
        #     provider="huggingface",
        #     api_key_env="HUGGINGFACE_API_KEY",
        #     rate_limit=30,
        #     timeout=90,
        #     additional_params={"type": "hf_router_api"}
        # ),
        # "bigcode/santacoder": ModelConfig(
        #     name="bigcode/santacoder",
        #     provider="huggingface",
        #     api_key_env="HUGGINGFACE_API_KEY",
        #     rate_limit=30,
        #     timeout=90,
        #     additional_params={"type": "hf_router_api"}
        # ),
        # Others
        #  "microsoft/phi-4": ModelConfig(
        #     name="microsoft/phi-4",
        #     provider="huggingface",
        #     api_key_env="HUGGINGFACE_API_KEY",
        #     rate_limit=30,
        #     timeout=90,
        #     additional_params={"type": "hf_router_api"}
        # ),
        
    }
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of all available model names"""
        return list(cls.DEFAULT_CONFIGS.keys())
    
    @classmethod
    def get_enabled_models(cls) -> List[ModelConfig]:
        """Get list of enabled models with valid API keys"""
        enabled_models = []
        
        for model_name, config in cls.DEFAULT_CONFIGS.items():
            if not config.enabled:
                continue
                
            # Check if API key is available
            api_key = os.getenv(config.api_key_env)
            if api_key:
                enabled_models.append(config)
            else:
                print(f"Warning: {config.api_key_env} not found for {model_name}")
        
        return enabled_models
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return cls.DEFAULT_CONFIGS.get(model_name)
    
    @classmethod
    def get_models_by_provider(cls, provider: str) -> List[ModelConfig]:
        """Get all models for a specific provider"""
        return [config for config in cls.DEFAULT_CONFIGS.values() 
                if config.provider == provider and config.enabled]
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Validate that required API keys are present"""
        key_status = {}
        
        for config in cls.DEFAULT_CONFIGS.values():
            if config.enabled:
                api_key = os.getenv(config.api_key_env)
                key_status[config.api_key_env] = api_key is not None
        
        return key_status
    
    @classmethod
    def get_extraction_config(cls) -> Optional[str]:
        """Get API key for solution extraction (defaults to OpenAI)"""
        return os.getenv("OPENAI_API_KEY")  # Use OpenAI for extraction by default
    
    @classmethod
    def create_models_from_config(cls, model_names: Optional[List[str]] = None):
        """
        Create model instances from configuration
        
        Args:
            model_names: List of specific models to create (None for all enabled)
            
        Returns:
            List of instantiated model objects
        """
        from ..models import (OpenAIModel, AnthropicModel, GoogleModel, 
                             XAIModel, MetaModel, DeepSeekModel)
        from ..models.huggingface_model import HuggingFaceRouterInferenceModel, HuggingFaceDirectInferenceModel, HuggingFaceLocalModel
        from ..models.cohere_model import CohereModel
        
        models = []
        configs_to_use = cls.get_enabled_models()
        
        # Filter by specific model names if provided (match against config keys)
        if model_names:
            # Get configs for the specified model keys
            specified_configs = []
            for model_key in model_names:
                config = cls.get_model_config(model_key)
                if config and config.enabled:
                    specified_configs.append(config)
            configs_to_use = specified_configs
        
        for config in configs_to_use:
            api_key = os.getenv(config.api_key_env)
            if not api_key:
                print(f"Skipping {config.name}: {config.api_key_env} not set")
                continue
            
            try:
                if config.provider == "openai":
                    model = OpenAIModel(
                        model_name=config.name,
                        api_key=api_key,
                        rate_limit=config.rate_limit,
                        timeout=config.timeout
                    )
                elif config.provider == "anthropic":
                    model = AnthropicModel(
                        model_name=config.name,
                        api_key=api_key,
                        rate_limit=config.rate_limit,
                        timeout=config.timeout
                    )
                elif config.provider == "google":
                    model = GoogleModel(
                        model_name=config.name,
                        api_key=api_key,
                        rate_limit=config.rate_limit,
                        timeout=config.timeout
                    )
                elif config.provider == "xai":
                    model = XAIModel(
                        model_name=config.name,
                        api_key=api_key,
                        rate_limit=config.rate_limit,
                        timeout=config.timeout
                    )
                elif config.provider == "meta":
                    provider_type = config.additional_params.get("provider", "together")
                    model = MetaModel(
                        model_name=config.name,
                        api_key=api_key,
                        provider=provider_type,
                        rate_limit=config.rate_limit,
                        timeout=config.timeout
                    )
                elif config.provider == "cohere":
                    model = CohereModel(
                        model_name=config.name,
                        api_key=api_key,
                        rate_limit=config.rate_limit,
                        timeout=config.timeout
                    )
                elif config.provider == "deepseek":
                    model = DeepSeekModel(
                        model_name=config.name,
                        api_key=api_key,
                        rate_limit=config.rate_limit,
                        timeout=config.timeout
                    )
                elif config.provider == "huggingface":
                    model_type = config.additional_params.get("type", "hf_router_api")
                    if model_type == "hf_router_api":
                        model = HuggingFaceRouterInferenceModel(
                            model_name=config.name,
                            api_key=api_key,
                            rate_limit=config.rate_limit,
                            timeout=config.timeout
                        )
                    elif model_type == "hf_direct_api":
                        model = HuggingFaceDirectInferenceModel(
                            model_name=config.name,
                            api_key=api_key,
                            rate_limit=config.rate_limit,
                            timeout=config.timeout
                        )
                    else:  # local
                        model = HuggingFaceLocalModel(
                            model_name=config.name,
                            api_key=api_key,
                            rate_limit=config.rate_limit,
                            timeout=config.timeout
                        )
                else:
                    print(f"Unknown provider {config.provider} for {config.name}")
                    continue
                
                models.append(model)
                print(f"Initialized {config.name}")
                
            except Exception as e:
                print(f"Failed to initialize {config.name}: {e}")
        
        return models