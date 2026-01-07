import os
from dotenv import load_dotenv
import gc
import time
import torch
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from threading import Lock

# HuggingFace imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.config_loader import ModelConfig, get_config_loader, AttackVariation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import document loader for RAG-based attacks
try:
    from src.document_loader import get_document_loader
    DOCUMENT_LOADER_AVAILABLE = True
except ImportError:
    DOCUMENT_LOADER_AVAILABLE = False
    logger.warning("Document loader not available")


@dataclass
class InferenceResult:
    """
    Result of a single inference call.

    Attributes:
        model_key: Key of the model used
        prompt: Input prompt sent to the model
        response: Generated response text
        tokens_generated: Number of tokens in the response
        response_time: Time taken for inference in seconds
        success: Whether inference completed without errors
        error_message: Error message if inference failed
    """
    model_key: str
    prompt: str
    response: str
    tokens_generated: int
    response_time: float
    success: bool
    error_message: Optional[str] = None


class ModelManager:
    """
    Manages loading and caching of LLM models.

    This class handles:
    - Loading models with 4-bit quantization
    - GPU/CPU device detection and allocation
    - Model caching for faster subsequent loads
    - Memory cleanup when switching models

    Design Notes:
    - Only one model is loaded at a time to conserve memory
    - Models are cached after first load for faster switching
    - Automatic fallback to CPU if CUDA is unavailable
    """

    def __init__(self):
        """Initialize the model manager."""
        self.config_loader = get_config_loader()
        self.current_model_key: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self._lock = Lock()
        # Load environment variables (e.g., HUGGINGFACE token)
        load_dotenv()
        # Support several common env var names for HuggingFace token
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')

        # Track models that need use_cache=False (e.g., Phi-3.5 with DynamicCache issues)
        self._models_needing_no_cache: set = set()

        # Detect available device
        self.device = self._detect_device()
        logger.info(f"Model Manager initialized. Device: {self.device}")

    def _detect_device(self) -> str:
        """
        Detect the best available device for inference.

        Returns:
            'cuda' if NVIDIA GPU with CUDA is available, 'cpu' otherwise

        Notes:
            - Checks for CUDA availability via PyTorch
            - Logs GPU information if available
            - Falls back gracefully to CPU
        """
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
            return "cuda"
        else:
            logger.warning(
                "CUDA not available, using CPU (inference will be slower)")
            return "cpu"

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Create the 4-bit quantization configuration.

        Returns:
            BitsAndBytesConfig for 4-bit quantization, or None if not supported

        Notes:
            - Uses NF4 (Normalized Float 4-bit) quantization
            - Double quantization for memory efficiency
            - FP16 compute dtype for best performance
        """
        # Skip quantization on CPU (bitsandbytes requires CUDA)
        if self.device != "cuda":
            logger.info("Skipping 4-bit quantization (CPU mode)")
            return None

        try:
            quant_config = self.config_loader.get_quantization_config()

            return BitsAndBytesConfig(
                load_in_4bit=quant_config.get('load_in_4bit', True),
                bnb_4bit_compute_dtype=getattr(
                    torch,
                    quant_config.get('bnb_4bit_compute_dtype', 'float16')
                ),
                bnb_4bit_quant_type=quant_config.get(
                    'bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=quant_config.get(
                    'bnb_4bit_use_double_quant', True)
            )
        except ImportError:
            logger.warning(
                "bitsandbytes not available, loading model without quantization")
            return None

    def _cleanup_memory(self):
        """
        Clean up GPU/CPU memory from previously loaded models.

        This is critical for systems with limited RAM to ensure
        only one model occupies memory at a time.
        """
        if self.model is not None:
            logger.info(f"Unloading model: {self.current_model_key}")
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.current_model_key = None

    def load_model(self, model_key: str, force_reload: bool = False) -> Tuple[bool, str]:
        """
        Load a model by its configuration key.

        Args:
            model_key: The model key
            force_reload: If True, reload even if model is already loaded

        Returns:
            Tuple of (success: bool, message: str)

        Notes:
            - Thread-safe via locking
            - Automatically cleans up previous model
            - Downloads model if not cached locally
        """
        with self._lock:
            # Check if already loaded
            if self.current_model_key == model_key and not force_reload:
                logger.info(f"Model {model_key} already loaded")
                return True, f"Model {model_key} already loaded"

            # Get model configuration
            model_config = self.config_loader.get_model(model_key)
            if model_config is None:
                return False, f"Model configuration not found: {model_key}"

            # Cleanup previous model
            self._cleanup_memory()

            logger.info(
                f"Loading model: {model_config.name} ({model_config.model_id})")

            try:
                # Get quantization config
                quant_config = self._get_quantization_config()

                # Load tokenizer
                logger.info("Loading tokenizer...")
                tokenizer_kwargs = {
                    "trust_remote_code": True,
                    "padding_side": 'left'
                }
                if self.hf_token:
                    tokenizer_kwargs["token"] = self.hf_token

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_config.model_id,
                    **tokenizer_kwargs
                )

                # Set pad token if not defined
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Load model with quantization
                logger.info("Loading model (this may take a few minutes)...")

                model_kwargs = {
                    "trust_remote_code": True,
                    "device_map": "auto" if self.device == "cuda" else None,
                    "low_cpu_mem_usage": True,
                }

                if quant_config is not None:
                    model_kwargs["quantization_config"] = quant_config
                elif self.device == "cuda":
                    model_kwargs["dtype"] = torch.float16

                if self.hf_token:
                    model_kwargs["token"] = self.hf_token

                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_config.model_id,
                        **model_kwargs
                    )
                except (ValueError, OSError) as e:
                    # Handle specific model loading errors
                    error_str = str(e)
                    if "Unrecognized configuration class" in error_str or "not found in" in error_str:
                        raise ValueError(
                            f"Model '{model_config.model_id}' is not compatible with AutoModelForCausalLM. "
                            f"This may be a multimodal model or require a different model class. "
                            f"Please update config/models.yaml with a text-only model."
                        )
                    raise  # Re-raise if it's a different error

                # Move to CPU if not using CUDA device_map
                if self.device == "cpu":
                    self.model = self.model.to("cpu")

                self.current_model_key = model_key

                # Log memory usage
                if self.device == "cuda":
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    logger.info(
                        f"Model loaded. GPU memory used: {memory_used:.2f}GB")

                return True, f"Successfully loaded {model_config.name}"

            except Exception as e:
                self._cleanup_memory()
                error_msg = f"Failed to load model {model_key}: {str(e)}"
                logger.error(error_msg)
                return False, error_msg

    def get_current_model_config(self) -> Optional[ModelConfig]:
        """Get the configuration of the currently loaded model."""
        if self.current_model_key is None:
            return None
        return self.config_loader.get_model(self.current_model_key)

    def is_model_loaded(self) -> bool:
        """Check if any model is currently loaded."""
        return self.model is not None and self.tokenizer is not None


class InferenceEngine:
    """
    Handles text generation using loaded LLM models.

    This class provides:
    - Text generation with configurable parameters
    - Proper prompt formatting for different model architectures
    - Defense prompt integration for security testing
    - Response time and token count tracking
    - Error handling and graceful degradation
    """

    def __init__(self, model_manager: ModelManager):
        """
        Initialize the inference engine.

        Args:
            model_manager: The ModelManager instance for model access
        """
        self.model_manager = model_manager
        self.config_loader = get_config_loader()

    def _format_prompt(self, user_prompt: str, model_config: ModelConfig, use_defense: bool = True) -> tuple:
        """
        Format the prompt according to the model's expected template.

        Args:
            user_prompt: The user's input prompt (attack prompt)
            model_config: Configuration of the target model
            use_defense: Whether to apply defense prompt mechanisms

        Returns:
            Tuple of (formatted_prompt_string, messages_list) where:
            - formatted_prompt_string: String formatted with template (fallback)
            - messages_list: List of message dicts for apply_chat_template

        Notes:
            - Different models expect different chat formats
            - System prompt is included from model config
            - Template includes proper special tokens
            - Defense prompts are applied when enabled
            - Returns both string and messages for flexibility
        """
        from src.defense_prompts import get_defense_builder

        template = self.config_loader.get_prompt_template(
            model_config.prompt_template)

        # Apply defense mechanisms if enabled
        if use_defense:
            defense_builder = get_defense_builder()
            system_prompt = defense_builder.build_system_prompt(
                model_config.system_prompt.strip())
            user_prompt_formatted = defense_builder.wrap_user_prompt(
                user_prompt.strip())
        else:
            system_prompt = model_config.system_prompt.strip()
            user_prompt_formatted = user_prompt.strip()

        # Create messages list for apply_chat_template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_formatted}
        ]

        # Also create string formatted version as fallback
        formatted_string = template.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt_formatted
        )

        return formatted_string, messages

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        use_system_prompt: bool = True,
        use_defense: bool = True,
        attack: Optional[AttackVariation] = None
    ) -> InferenceResult:
        """
        Generate a response for the given prompt.

        Args:
            prompt: The input prompt (attack prompt to test)
            max_new_tokens: Maximum tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)
            top_p: Top-p sampling threshold (overrides config)
            use_system_prompt: Whether to include the model's system prompt
            use_defense: Whether to apply defense prompt mechanisms
            attack: Optional AttackVariation object for document injection attacks

        Returns:
            InferenceResult containing response and metrics

        Notes:
            - Measures response time for metrics
            - Counts generated tokens
            - Handles errors gracefully with informative messages
            - Applies defense prompts when use_defense is True
            - Loads documents from files for document injection attacks
        """
        # Load document if this is a document injection attack
        full_prompt = prompt
        if attack and attack.document_file and DOCUMENT_LOADER_AVAILABLE:
            attack_settings = self.config_loader.get_attack_settings()
            use_document_files = attack_settings.get(
                'use_document_files', True)

            if use_document_files:
                doc_loader = get_document_loader()
                document_content = doc_loader.load_document(
                    attack.document_file)

                if document_content:
                    # Combine prompt with document content
                    full_prompt = f"{prompt}\n\n---DOCUMENT---\n{document_content}\n---END DOCUMENT---"
                    logger.info(
                        f"Loaded document '{attack.document_file}' for attack '{attack.id}' ({len(document_content)} chars)")
                else:
                    logger.warning(
                        f"Failed to load document '{attack.document_file}' for attack '{attack.id}'")

        # Check if model is loaded
        if not self.model_manager.is_model_loaded():
            return InferenceResult(
                model_key="none",
                prompt=full_prompt,
                response="",
                tokens_generated=0,
                response_time=0.0,
                success=False,
                error_message="No model loaded"
            )

        model_config = self.model_manager.get_current_model_config()
        gen_config = model_config.generation_config

        # Format prompt with optional defense mechanisms
        if use_system_prompt:
            formatted_prompt, messages = self._format_prompt(
                full_prompt, model_config, use_defense=use_defense)
        else:
            formatted_prompt = full_prompt
            messages = [{"role": "user", "content": full_prompt}]

        # Try to use apply_chat_template with enable_thinking=False for models that support it
        # This prevents thinking mode tokens from being generated (better than post-processing)
        try:
            # Check if model supports chat template
            if hasattr(self.model_manager.tokenizer, 'apply_chat_template'):
                # Try applying chat template with system role first
                try:
                    formatted_prompt = self.model_manager.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False  # Disable thinking mode at tokenizer level
                    )
                    logger.debug(
                        f"Applied chat template with enable_thinking=False")
                except Exception as template_error:
                    error_msg = str(template_error).lower()
                    # Handle models that don't support system role (e.g., Gemma 2)
                    if "system role not supported" in error_msg or "system" in error_msg:
                        logger.info(
                            "Model doesn't support system role, merging into user message")
                        # Merge system prompt into user message
                        system_content = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
                        user_content = messages[-1]["content"] if messages else ""

                        if system_content:
                            merged_content = f"[System Instructions]\n{system_content}\n\n[User Query]\n{user_content}"
                        else:
                            merged_content = user_content

                        messages_no_system = [
                            {"role": "user", "content": merged_content}]

                        try:
                            formatted_prompt = self.model_manager.tokenizer.apply_chat_template(
                                messages_no_system,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False
                            )
                        except TypeError:
                            # enable_thinking not supported
                            formatted_prompt = self.model_manager.tokenizer.apply_chat_template(
                                messages_no_system,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                    else:
                        # Re-raise if it's a different error
                        raise template_error
        except TypeError as e:
            # Model doesn't support enable_thinking parameter, try without it
            logger.debug(
                f"Model doesn't support enable_thinking parameter: {e}")
            try:
                formatted_prompt = self.model_manager.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as template_error:
                error_msg = str(template_error).lower()
                if "system role not supported" in error_msg or "system" in error_msg:
                    logger.info(
                        "Model doesn't support system role, merging into user message")
                    system_content = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
                    user_content = messages[-1]["content"] if messages else ""

                    if system_content:
                        merged_content = f"[System Instructions]\n{system_content}\n\n[User Query]\n{user_content}"
                    else:
                        merged_content = user_content

                    messages_no_system = [
                        {"role": "user", "content": merged_content}]
                    formatted_prompt = self.model_manager.tokenizer.apply_chat_template(
                        messages_no_system,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    raise template_error
        except ValueError as e:
            logger.debug(f"Chat template error: {e}")
            pass

        # Prepare generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or gen_config.get('max_new_tokens', 512),
            "temperature": temperature or gen_config.get('temperature', 0.1),
            "do_sample": gen_config.get('do_sample', False),
            "repetition_penalty": gen_config.get('repetition_penalty', 1.1),
            "pad_token_id": self.model_manager.tokenizer.pad_token_id,
            "eos_token_id": self.model_manager.tokenizer.eos_token_id,
        }

        # Check if this model needs use_cache=False (e.g., due to DynamicCache issues)
        current_model = self.model_manager.current_model_key
        if current_model and current_model in self.model_manager._models_needing_no_cache:
            gen_kwargs['use_cache'] = False
            logger.debug(f"Using use_cache=False for model {current_model}")

        # Ensure temperature is not too low when sampling (prevents numerical instability)
        # Temperature 0.0 is allowed when do_sample=False (greedy decoding)
        if gen_kwargs['do_sample'] and gen_kwargs['temperature'] < 0.1:
            gen_kwargs['temperature'] = 0.1
            logger.warning(
                "Temperature too low for sampling mode, set to 0.1 to prevent numerical instability")

        try:
            # Tokenize input
            inputs = self.model_manager.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )

            # Move inputs to the correct device
            if self.model_manager.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            input_length = inputs['input_ids'].shape[1]

            # Generate response with timing
            start_time = time.time()

            # Attempt generation with retry on numerical instability
            max_retries = 3
            last_error = None

            for attempt in range(max_retries):
                try:
                    with torch.no_grad():
                        outputs = self.model_manager.model.generate(
                            **inputs,
                            **gen_kwargs
                        )
                    break  # Success, exit retry loop

                except RuntimeError as e:
                    error_str = str(e)
                    last_error = e

                    # Check for probability tensor errors (numerical instability)
                    if "probability tensor contains" in error_str or "nan" in error_str.lower():
                        if attempt < max_retries - 1:
                            # Adjust parameters for next retry
                            gen_kwargs['temperature'] = min(
                                gen_kwargs['temperature'] + 0.2, 1.0)
                            if 'top_k' not in gen_kwargs:
                                gen_kwargs['top_k'] = 50
                            logger.warning(
                                f"Numerical instability detected (attempt {attempt + 1}/{max_retries}). "
                                f"Retrying with temperature={gen_kwargs['temperature']:.2f}"
                            )
                            continue
                    # If not a probability error or last retry, re-raise
                    raise

                except AttributeError as e:
                    error_str = str(e)
                    last_error = e

                    # Handle DynamicCache compatibility issue (affects Phi-3.5 and some other models)
                    if "DynamicCache" in error_str or "seen_tokens" in error_str:
                        if attempt < max_retries - 1 and gen_kwargs.get('use_cache') is not False:
                            # Remember this model needs use_cache=False for future calls
                            current_model = self.model_manager.current_model_key
                            if current_model:
                                self.model_manager._models_needing_no_cache.add(
                                    current_model)
                                logger.info(
                                    f"Model '{current_model}' added to no-cache list for future calls"
                                )
                            logger.warning(
                                f"DynamicCache compatibility issue detected. "
                                f"Retrying with use_cache=False (attempt {attempt + 1}/{max_retries})"
                            )
                            gen_kwargs['use_cache'] = False
                            continue
                    # If not a cache error or last retry, re-raise
                    raise
            else:
                # All retries exhausted
                raise last_error

            response_time = time.time() - start_time

            # Decode response (exclude input tokens)
            response_tokens = outputs[0][input_length:]
            response_text = self.model_manager.tokenizer.decode(
                response_tokens,
                skip_special_tokens=True
            )

            # Note: Thinking mode is now disabled at tokenizer level via enable_thinking=False
            # No need for post-processing to strip <think> tags

            tokens_generated = len(response_tokens)

            return InferenceResult(
                model_key=self.model_manager.current_model_key,
                prompt=full_prompt,
                response=response_text.strip(),
                tokens_generated=tokens_generated,
                response_time=response_time,
                success=True
            )

        except Exception as e:
            error_str = str(e)

            # Provide more helpful error messages
            if "probability tensor contains" in error_str:
                error_msg = (
                    f"Numerical instability in model generation. This may be due to:\n"
                    f"  - Model incompatibility with current settings\n"
                    f"  - Temperature too low (current: {gen_kwargs.get('temperature', 'N/A')})\n"
                    f"  - Sampling parameters causing NaN/Inf values\n"
                    f"Original error: {error_str}"
                )
            elif "CUDA out of memory" in error_str:
                error_msg = (
                    f"GPU out of memory. Try:\n"
                    f"  - Using a smaller model\n"
                    f"  - Reducing max_new_tokens\n"
                    f"  - Restarting the application"
                )
            elif "DynamicCache" in error_str or "seen_tokens" in error_str:
                error_msg = (
                    f"Model cache compatibility issue (DynamicCache). This is a known issue with "
                    f"some models (e.g., Phi-3.5) and certain transformers versions.\n"
                    f"Try upgrading transformers: pip install --upgrade transformers\n"
                    f"Original error: {error_str}"
                )
            elif "System role not supported" in error_str:
                error_msg = (
                    f"This model doesn't support system prompts in its chat template.\n"
                    f"The system prompt will be merged with user message automatically.\n"
                    f"Original error: {error_str}"
                )
            else:
                error_msg = error_str

            logger.error(f"Inference error: {error_msg}")
            return InferenceResult(
                model_key=self.model_manager.current_model_key or "unknown",
                prompt=full_prompt,
                response="",
                tokens_generated=0,
                response_time=0.0,
                success=False,
                error_message=error_msg
            )


_model_manager: Optional[ModelManager] = None
_inference_engine: Optional[InferenceEngine] = None


def get_model_manager() -> ModelManager:
    """Get the global ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def get_inference_engine() -> InferenceEngine:
    """Get the global InferenceEngine instance."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine(get_model_manager())
    return _inference_engine


def load_model(model_key: str) -> Tuple[bool, str]:
    """Convenience function to load a model."""
    return get_model_manager().load_model(model_key)


def generate_response(prompt: str, **kwargs) -> InferenceResult:
    """Convenience function to generate a response."""
    return get_inference_engine().generate(prompt, **kwargs)
