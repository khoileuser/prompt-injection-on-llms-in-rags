import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for a single LLM model.

    Attributes:
        key: Unique identifier for the model
        name: Display name for UI
        model_id: HuggingFace model identifier
        description: Brief description of the model
        parameters: Model size in billions of parameters
        prompt_template: Name of the prompt template to use
        system_prompt: Default system prompt for the model
        generation_config: Dictionary of generation parameters
    """
    key: str
    name: str
    model_id: str
    description: str
    parameters: float
    prompt_template: str
    system_prompt: str
    generation_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackVariation:
    """
    A single attack prompt variation.

    Attributes:
        id: Unique identifier (e.g., 'direct_io_001')
        name: Human-readable name
        prompt: The actual attack prompt text
        description: Detailed explanation of the attack
        document_file: Optional filename for document injection attacks
        injection_vector: 'direct' or 'indirect'
        attack_objective: Attack goal
        success_definition: What constitutes a successful attack
    """
    id: str
    name: str
    prompt: str
    description: str
    document_file: Optional[str] = None
    injection_vector: str = "direct"  # 'direct' or 'indirect'
    attack_objective: str = "instruction_override"  # attack objective
    success_definition: str = ""  # explicit success criteria


@dataclass
class AttackCategory:
    """
    A category of attacks with multiple variations.

    Attributes:
        key: Unique identifier for the category
        category: Display name
        description: Detailed explanation of this attack type
        variations: List of attack variations
    """
    key: str
    category: str
    description: str
    variations: List[AttackVariation]


class ConfigLoader:
    """
    Handles loading and parsing of YAML configuration files.

    This class provides methods to load model configurations, attack
    configurations, and prompt templates from YAML files located in
    the config directory.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration loader.

        Args:
            config_dir: Path to the configuration directory. If None,
                       uses the default 'config' directory relative to
                       the project root.
        """
        if config_dir is None:
            # Default to config directory relative to this file
            self.config_dir = Path(__file__).parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)

        logger.info(f"Configuration directory: {self.config_dir}")

        # Cache for loaded configurations
        self._models_config: Optional[Dict] = None
        self._attacks_config: Optional[Dict] = None

    def _load_yaml(self, filename: str) -> Dict:
        """
        Load and parse a YAML file.

        Args:
            filename: Name of the YAML file to load

        Returns:
            Parsed YAML content as a dictionary

        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the file contains invalid YAML
        """
        filepath = self.config_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML file {filepath}: {e}")
                raise

    def load_models_config(self, force_reload: bool = False) -> Dict:
        """
        Load the models configuration file.

        Args:
            force_reload: If True, reload from disk even if cached

        Returns:
            Dictionary containing all model configurations
        """
        if self._models_config is None or force_reload:
            self._models_config = self._load_yaml("models.yaml")
            logger.info(
                f"Loaded {len(self._models_config.get('models', {}))} model configurations")

        return self._models_config

    def load_attacks_config(self, force_reload: bool = False) -> Dict:
        """
        Load the attacks configuration file.

        Args:
            force_reload: If True, reload from disk even if cached

        Returns:
            Dictionary containing all attack configurations
        """
        if self._attacks_config is None or force_reload:
            self._attacks_config = self._load_yaml("attacks.yaml")

            # Count total attack variations
            total_attacks = self._count_attacks(self._attacks_config)
            logger.info(f"Loaded {total_attacks} attack variations")

        return self._attacks_config

    def _count_attacks(self, config: Dict) -> int:
        """Count total attack variations in config."""
        total = 0
        # Count from taxonomy structure (direct_attacks, indirect_attacks)
        for vector_key in ['direct_attacks', 'indirect_attacks']:
            if vector_key in config:
                for objective, attacks in config[vector_key].items():
                    if isinstance(attacks, list):
                        total += len(attacks)
        # Count from legacy structure
        for key, category in config.items():
            if key not in ['settings', 'secrets', 'direct_attacks', 'indirect_attacks', 'legacy_mapping']:
                if isinstance(category, dict) and 'variations' in category:
                    total += len(category.get('variations', []))
        return total

    def get_models(self) -> List[ModelConfig]:
        """
        Get all model configurations as ModelConfig objects.

        Returns:
            List of ModelConfig objects for each configured model
        """
        config = self.load_models_config()
        models = []

        for key, model_data in config.get('models', {}).items():
            models.append(ModelConfig(
                key=key,
                name=model_data.get('name', key),
                model_id=model_data.get('model_id', ''),
                description=model_data.get('description', ''),
                parameters=model_data.get('parameters', 0),
                prompt_template=model_data.get('prompt_template', 'default'),
                system_prompt=model_data.get('system_prompt', ''),
                generation_config=model_data.get('generation_config', {})
            ))

        return models

    def get_model(self, key: str) -> Optional[ModelConfig]:
        """
        Get a specific model configuration by key.

        Args:
            key: The model key (e.g., 'llama', 'deepseek')

        Returns:
            ModelConfig object or None if not found
        """
        models = self.get_models()
        for model in models:
            if model.key == key:
                return model
        return None

    def get_attack_categories(self) -> List[AttackCategory]:
        """
        Get all attack categories with their variations.
        Supports both legacy and matrix-based configurations.

        Returns:
            List of AttackCategory objects
        """
        config = self.load_attacks_config()
        categories = []

        # Check if using taxonomy structure
        if 'direct_attacks' in config or 'indirect_attacks' in config:
            return self._get_taxonomy_categories(config)

        # Legacy attack category keys (exclude 'settings')
        category_keys = [
            'instruction_override',
            'data_extraction',
            'role_playing',
            'document_injection',
            'code_injection'
        ]

        for key in category_keys:
            if key in config:
                cat_data = config[key]
                variations = []

                for var_data in cat_data.get('variations', []):
                    variations.append(AttackVariation(
                        id=var_data.get('id', ''),
                        name=var_data.get('name', ''),
                        prompt=var_data.get('prompt', ''),
                        description=var_data.get('description', ''),
                        document_file=var_data.get('document_file', None),
                        injection_vector=var_data.get(
                            'injection_vector', 'direct'),
                        attack_objective=var_data.get('attack_objective', key),
                        success_definition=var_data.get(
                            'success_definition', '')
                    ))

                categories.append(AttackCategory(
                    key=key,
                    category=cat_data.get('category', key),
                    description=cat_data.get('description', ''),
                    variations=variations
                ))

        return categories

    def _get_taxonomy_categories(self, config: Dict) -> List[AttackCategory]:
        """
        Parse taxonomy-based attack configuration.

        Returns:
            List of AttackCategory objects organized by taxonomy
        """
        categories = []

        # Process direct attacks
        if 'direct_attacks' in config:
            for objective, attacks in config['direct_attacks'].items():
                if isinstance(attacks, list):
                    variations = []
                    for var_data in attacks:
                        variations.append(AttackVariation(
                            id=var_data.get('id', ''),
                            name=var_data.get('name', ''),
                            prompt=var_data.get('prompt', ''),
                            description=var_data.get('description', ''),
                            document_file=var_data.get('document_file', None),
                            injection_vector='direct',
                            attack_objective=objective,
                            success_definition=var_data.get(
                                'success_definition', '')
                        ))

                    categories.append(AttackCategory(
                        key=f"direct_{objective}",
                        category=f"Direct {objective.replace('_', ' ').title()}",
                        description=f"Direct injection attacks targeting {objective}",
                        variations=variations
                    ))

        # Process indirect attacks
        if 'indirect_attacks' in config:
            for objective, attacks in config['indirect_attacks'].items():
                if isinstance(attacks, list):
                    variations = []
                    for var_data in attacks:
                        variations.append(AttackVariation(
                            id=var_data.get('id', ''),
                            name=var_data.get('name', ''),
                            prompt=var_data.get('prompt', ''),
                            description=var_data.get('description', ''),
                            document_file=var_data.get('document_file', None),
                            injection_vector='indirect',
                            attack_objective=objective,
                            success_definition=var_data.get(
                                'success_definition', '')
                        ))

                    categories.append(AttackCategory(
                        key=f"indirect_{objective}",
                        category=f"Indirect {objective.replace('_', ' ').title()}",
                        description=f"Indirect injection attacks via documents targeting {objective}",
                        variations=variations
                    ))

        return categories

    def get_attacks_by_taxonomy(self) -> Dict[str, Dict[str, List[AttackVariation]]]:
        """
        Get attacks organized by matrix (vector × objective).

        Returns:
            Nested dict: {injection_vector: {attack_objective: [attacks]}}
        """
        config = self.load_attacks_config()
        taxonomy = {
            'direct': {
                'instruction_override': [],
                'data_extraction': [],
                'role_confusion': []
            },
            'indirect': {
                'instruction_override': [],
                'data_extraction': [],
                'role_confusion': []
            }
        }

        # Process taxonomy-structured config
        if 'direct_attacks' in config:
            for objective, attacks in config['direct_attacks'].items():
                if isinstance(attacks, list) and objective in taxonomy['direct']:
                    for var_data in attacks:
                        taxonomy['direct'][objective].append(AttackVariation(
                            id=var_data.get('id', ''),
                            name=var_data.get('name', ''),
                            prompt=var_data.get('prompt', ''),
                            description=var_data.get('description', ''),
                            document_file=var_data.get('document_file', None),
                            injection_vector='direct',
                            attack_objective=objective,
                            success_definition=var_data.get(
                                'success_definition', '')
                        ))

        if 'indirect_attacks' in config:
            for objective, attacks in config['indirect_attacks'].items():
                if isinstance(attacks, list) and objective in taxonomy['indirect']:
                    for var_data in attacks:
                        taxonomy['indirect'][objective].append(AttackVariation(
                            id=var_data.get('id', ''),
                            name=var_data.get('name', ''),
                            prompt=var_data.get('prompt', ''),
                            description=var_data.get('description', ''),
                            document_file=var_data.get('document_file', None),
                            injection_vector='indirect',
                            attack_objective=objective,
                            success_definition=var_data.get(
                                'success_definition', '')
                        ))

        return taxonomy

    def get_secrets(self) -> Dict[str, str]:
        """
        Get the secrets defined for data extraction attacks.

        Returns:
            Dictionary of secret_name: secret_value
        """
        config = self.load_attacks_config()
        return config.get('secrets', {})

    def get_attack_by_id(self, attack_id: str) -> Optional[AttackVariation]:
        """
        Get a specific attack variation by its ID.

        Args:
            attack_id: The attack ID (e.g., 'io_001', 'de_005')

        Returns:
            AttackVariation object or None if not found
        """
        categories = self.get_attack_categories()
        for category in categories:
            for variation in category.variations:
                if variation.id == attack_id:
                    return variation
        return None

    def get_all_attacks(self) -> List[AttackVariation]:
        """
        Get all attack variations across all categories.

        Returns:
            Flat list of all AttackVariation objects
        """
        categories = self.get_attack_categories()
        attacks = []
        for category in categories:
            attacks.extend(category.variations)
        return attacks

    def get_prompt_template(self, template_name: str) -> str:
        """
        Get a prompt template by name.

        Args:
            template_name: Name of the template (e.g., 'llama', 'qwen')

        Returns:
            Template string with {system_prompt} and {user_prompt} placeholders
        """
        config = self.load_models_config()
        templates = config.get('prompt_templates', {})
        return templates.get(template_name, "{system_prompt}\n\n{user_prompt}")

    def get_inference_settings(self) -> Dict[str, Any]:
        """
        Get global inference settings.

        Returns:
            Dictionary of inference settings (device_map, max_tokens, etc.)
        """
        config = self.load_models_config()
        return config.get('inference', {})

    def get_quantization_config(self) -> Dict[str, Any]:
        """
        Get 4-bit quantization configuration.

        Returns:
            Dictionary of quantization settings for bitsandbytes
        """
        config = self.load_models_config()
        return config.get('quantization', {})

    def get_attack_settings(self) -> Dict[str, Any]:
        """
        Get global attack settings.

        Returns:
            Dictionary of attack settings (include_rag_context, use_document_files, etc.)
        """
        config = self.load_attacks_config()
        return config.get('settings', {})


# Singleton instance for convenience
_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """
    Get the global configuration loader instance.

    Returns:
        The singleton ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


# Convenience functions
def get_models() -> List[ModelConfig]:
    """Get all model configurations."""
    return get_config_loader().get_models()


def get_attacks() -> List[AttackVariation]:
    """Get all attack variations."""
    return get_config_loader().get_all_attacks()


def get_attack_categories() -> List[AttackCategory]:
    """Get all attack categories."""
    return get_config_loader().get_attack_categories()
