# Bootstrap Refactor Implementation Assignment

**For:** Future Claude Code Agent
**Project:** multi-llm-chat Bootstrap Mechanism Refactor
**Estimated Duration:** 15-20 hours across 3 weeks
**Difficulty:** Advanced (requires careful refactoring with backward compatibility)

---

## Assignment Overview

You are tasked with refactoring the multi-llm-chat bootstrap mechanism by breaking apart the monolithic `src/ai_battle.py` file (2,243 lines) into a clean, modular architecture. This refactor must preserve 100% of existing functionality while improving code organization, testability, and maintainability.

**Primary Goal:** Transform the codebase from a single monolithic file to a well-organized library with multiple focused entry points.

**Success Criteria:**
1. All existing functionality works identically
2. Existing UI (`src/ui.py`) continues to work without changes
3. All example configurations continue to work
4. Code is more maintainable (no file > 500 lines)
5. Entry points are configuration-driven
6. Test coverage for all new modules

---

## Pre-Implementation Requirements

### 1. Read and Understand These Files
Before starting, thoroughly read and understand:

**Core files to analyze:**
- `/home/user/multi-llm-chat/src/ai_battle.py` - The monolithic file to refactor
- `/home/user/multi-llm-chat/docs/examples/run_vision_discussion.py` - Good pattern to follow
- `/home/user/multi-llm-chat/BOOTSTRAP_REFACTOR_PROPOSAL.md` - Architectural proposal
- `/home/user/multi-llm-chat/src/configuration.py` - Configuration loading
- `/home/user/multi-llm-chat/src/configdataclasses.py` - Data structures

**Example configurations:**
- `/home/user/multi-llm-chat/docs/examples/configs/vision_discussion.yaml`
- Other YAML files in `docs/examples/configs/`

### 2. Set Up Development Branch
```bash
# Create feature branch
git checkout -b refactor/bootstrap-mechanism

# Ensure clean working directory
git status
```

### 3. Create Backup
```bash
# Create backup of current ai_battle.py
cp src/ai_battle.py src/ai_battle.py.backup
```

---

## Phase 1: Extract Model Registry

**Duration:** 2-3 hours
**Goal:** Externalize model configurations from Python code to YAML

### Tasks

#### 1.1: Create Model Registry YAML
**File:** `src/data/model_registry.yaml`

**Action:** Extract model configuration dictionaries from `ai_battle.py` lines 112-255 and convert to YAML format.

**Source (from ai_battle.py):**
```python
OPENAI_MODELS = {
    "o1": {"model": "o1", "reasoning_level": "medium", "multimodal": False},
    "o3": {"model": "o3", "reasoning_level": "auto", "multimodal": True},
    # ... 18 models total
}

GEMINI_MODELS = {
    "gemini-2.0-pro": {"model": "gemini-2.0-pro-exp-02-05", "multimodal": True},
    # ... 9 models total
}

CLAUDE_MODELS = {
    "claude": {"model": "claude-3-7-sonnet-latest", "reasoning_level": None, "extended_thinking": False},
    # ... 21 models total
}

OLLAMA_THINKING_CONFIG = { ... }
```

**Target Structure:**
```yaml
# src/data/model_registry.yaml
openai_models:
  o1:
    model: "o1"
    reasoning_level: "medium"
    multimodal: false

  o3:
    model: "o3"
    reasoning_level: "auto"
    multimodal: true

  # ... continue for all 18 OpenAI models

gemini_models:
  gemini-2.0-pro:
    model: "gemini-2.0-pro-exp-02-05"
    multimodal: true

  # ... continue for all 9 Gemini models

claude_models:
  claude:
    model: "claude-3-7-sonnet-latest"
    reasoning_level: null
    extended_thinking: false

  # ... continue for all 21 Claude models

ollama_thinking_config:
  # ... migrate Ollama config
```

**Verification:**
- [ ] All OpenAI models migrated correctly
- [ ] All Gemini models migrated correctly
- [ ] All Claude models migrated correctly
- [ ] Ollama thinking config migrated
- [ ] YAML syntax is valid (`python -c "import yaml; yaml.safe_load(open('src/data/model_registry.yaml'))"`)

#### 1.2: Create Model Registry Module
**File:** `src/core/model_registry.py`

**Template:**
```python
"""Model registry for managing model configurations."""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for model configurations."""

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize registry from YAML file.

        Args:
            registry_path: Path to model registry YAML file.
                          Defaults to src/data/model_registry.yaml
        """
        if registry_path is None:
            # Default path relative to this file
            registry_path = Path(__file__).parent.parent / "data" / "model_registry.yaml"

        logger.debug(f"Loading model registry from: {registry_path}")

        with open(registry_path) as f:
            data = yaml.safe_load(f)

        self.openai_models = data.get("openai_models", {})
        self.gemini_models = data.get("gemini_models", {})
        self.claude_models = data.get("claude_models", {})
        self.ollama_thinking_config = data.get("ollama_thinking_config", {})

        logger.info(
            f"Loaded model registry: {len(self.openai_models)} OpenAI, "
            f"{len(self.gemini_models)} Gemini, {len(self.claude_models)} Claude models"
        )

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific model.

        Args:
            model_name: Name of the model (e.g., "o3", "claude-sonnet-4")

        Returns:
            Model configuration dictionary or None if not found
        """
        # Check all registries
        for registry in [self.openai_models, self.gemini_models, self.claude_models]:
            if model_name in registry:
                return registry[model_name]

        logger.warning(f"Model not found in registry: {model_name}")
        return None

    def list_models(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        List all models or models from a specific provider.

        Args:
            provider: Provider name ("openai", "gemini", "claude") or None for all

        Returns:
            Dictionary of model configurations
        """
        if provider == "openai":
            return self.openai_models
        elif provider == "gemini":
            return self.gemini_models
        elif provider == "claude":
            return self.claude_models
        elif provider is None:
            # Return all models
            return {
                **self.openai_models,
                **self.gemini_models,
                **self.claude_models,
            }
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def get_ollama_thinking_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get Ollama thinking configuration for a model.

        Args:
            model_name: Ollama model name

        Returns:
            Thinking configuration or None
        """
        return self.ollama_thinking_config.get(model_name)


# Singleton instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """
    Get the global model registry instance.

    Returns:
        Singleton ModelRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reload_registry(registry_path: Optional[Path] = None) -> ModelRegistry:
    """
    Reload the model registry from file.

    Args:
        registry_path: Optional path to registry file

    Returns:
        Reloaded ModelRegistry instance
    """
    global _registry
    _registry = ModelRegistry(registry_path)
    return _registry
```

**Verification:**
- [ ] Module imports without errors
- [ ] Can load registry: `registry = ModelRegistry()`
- [ ] Can get model config: `registry.get_model_config("o3")` returns correct dict
- [ ] Can list models: `registry.list_models("openai")` returns all OpenAI models
- [ ] Singleton works: `get_registry()` returns same instance

#### 1.3: Write Unit Tests
**File:** `tests/test_model_registry.py`

**Required Tests:**
```python
def test_load_registry():
    """Test loading registry from YAML."""
    pass

def test_get_model_config_openai():
    """Test getting OpenAI model configuration."""
    pass

def test_get_model_config_gemini():
    """Test getting Gemini model configuration."""
    pass

def test_get_model_config_claude():
    """Test getting Claude model configuration."""
    pass

def test_get_model_config_not_found():
    """Test handling of missing model."""
    pass

def test_list_models_all():
    """Test listing all models."""
    pass

def test_list_models_by_provider():
    """Test listing models by provider."""
    pass

def test_singleton_pattern():
    """Test registry singleton pattern."""
    pass
```

**Verification:**
- [ ] All tests pass: `pytest tests/test_model_registry.py -v`
- [ ] Test coverage > 90%: `pytest --cov=src.core.model_registry`

#### 1.4: Update Documentation
**File:** `README.md` or `docs/model_registry.md`

**Add section:**
```markdown
## Model Registry

Models are configured in `src/data/model_registry.yaml`. To add a new model:

1. Edit `src/data/model_registry.yaml`
2. Add model under appropriate provider section
3. Reload application

Example:
\`\`\`yaml
openai_models:
  my-new-model:
    model: "my-new-model-v1"
    reasoning_level: "medium"
    multimodal: true
\`\`\`
```

**Verification:**
- [ ] Documentation is clear and accurate
- [ ] Example code works

#### 1.5: Create Migration Stub in ai_battle.py
**Action:** Add backward compatibility imports at top of `ai_battle.py`

**Add after line 41:**
```python
# Import from new model registry (backward compatibility)
from core.model_registry import get_registry

_registry = get_registry()
OPENAI_MODELS = _registry.openai_models
GEMINI_MODELS = _registry.gemini_models
CLAUDE_MODELS = _registry.claude_models
OLLAMA_THINKING_CONFIG = _registry.ollama_thinking_config
```

**Verification:**
- [ ] `ai_battle.py` still works
- [ ] Can import models: `from ai_battle import OPENAI_MODELS`
- [ ] Models are loaded from YAML

**Commit Point:**
```bash
git add src/data/model_registry.yaml src/core/model_registry.py tests/test_model_registry.py
git commit -m "feat: Extract model registry to YAML configuration

- Create src/data/model_registry.yaml with all model configs
- Implement ModelRegistry class for loading and accessing models
- Add comprehensive unit tests
- Maintain backward compatibility in ai_battle.py

Part of bootstrap refactor (Phase 1)"
```

---

## Phase 2: Separate I/O Functions

**Duration:** 2-3 hours
**Goal:** Extract conversation saving and reporting functions

### Tasks

#### 2.1: Create Conversation I/O Module
**File:** `src/io/conversation_io.py`

**Action:** Extract these functions from `ai_battle.py`:
- `save_conversation()` (around line 1800-2000)
- `_sanitize_filename_part()` (utility function)
- `_render_signal_dashboard()` (visualization function)

**Template:**
```python
"""Conversation I/O operations for saving and loading conversations."""
import os
import base64
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


async def save_conversation(
    conversation: List[Dict[str, Any]],
    filename: str,
    human_model: str,
    ai_model: str,
    file_data: Optional[Any] = None,
    mode: str = "human-ai",
    signal_history: Optional[List] = None,
) -> str:
    """
    Save conversation to HTML file with embedded media.

    Args:
        conversation: List of conversation messages
        filename: Output filename
        human_model: Name of human model
        ai_model: Name of AI model
        file_data: Optional file data to embed (dict or list of dicts)
        mode: Conversation mode (human-ai, ai-ai, no-meta)
        signal_history: Optional signal history for visualization

    Returns:
        Path to saved file
    """
    # TODO: Copy implementation from ai_battle.py save_conversation()
    # Lines approximately 1800-2000

    logger.info(f"Saving conversation to: {filename}")

    # Implementation goes here...

    logger.info(f"Conversation saved successfully: {filename}")
    return os.path.abspath(filename)


def _sanitize_filename_part(part: str, max_length: int = 50) -> str:
    """
    Sanitize a part of a filename by removing invalid characters.

    Args:
        part: String to sanitize
        max_length: Maximum length of sanitized string

    Returns:
        Sanitized string safe for use in filenames
    """
    # TODO: Copy implementation from ai_battle.py
    pass


def _render_signal_dashboard(signal_history: List) -> str:
    """
    Render signal dashboard HTML for conversation context signals.

    Args:
        signal_history: List of signal data from conversation

    Returns:
        HTML string for signal visualization
    """
    # TODO: Copy implementation from ai_battle.py
    pass
```

**Implementation Notes:**
1. Search for `async def save_conversation` in `ai_battle.py`
2. Copy entire function body including all HTML generation logic
3. Ensure all imports are included (base64, datetime, etc.)
4. Copy helper functions `_sanitize_filename_part` and `_render_signal_dashboard`
5. Preserve exact functionality - this should be a pure move, not a rewrite

**Verification:**
- [ ] Module imports without errors
- [ ] Can call `save_conversation()` with test data
- [ ] Generated HTML file is valid
- [ ] File data embedding works (images, videos)

#### 2.2: Create Report Generator Module
**File:** `src/io/report_generator.py`

**Action:** Extract these functions from `ai_battle.py`:
- `save_arbiter_report()`
- `save_metrics_report()`

**Template:**
```python
"""Report generation for arbiter and metrics analysis."""
import os
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


async def save_arbiter_report(
    evaluation_result: Dict[str, Any],
    filename: str,
    human_model: str,
    ai_model: str,
) -> str:
    """
    Save arbiter evaluation report to HTML.

    Args:
        evaluation_result: Evaluation results from arbiter_v4
        filename: Output filename
        human_model: Name of human model
        ai_model: Name of AI model

    Returns:
        Path to saved file
    """
    # TODO: Copy implementation from ai_battle.py
    logger.info(f"Saving arbiter report to: {filename}")

    # Implementation goes here...

    logger.info(f"Arbiter report saved: {filename}")
    return os.path.abspath(filename)


async def save_metrics_report(
    metrics_result: Dict[str, Any],
    filename: str,
) -> str:
    """
    Save metrics analysis report to HTML.

    Args:
        metrics_result: Metrics analysis results from metrics_analyzer
        filename: Output filename

    Returns:
        Path to saved file
    """
    # TODO: Copy implementation from ai_battle.py
    logger.info(f"Saving metrics report to: {filename}")

    # Implementation goes here...

    logger.info(f"Metrics report saved: {filename}")
    return os.path.abspath(filename)
```

**Implementation Notes:**
1. Search for `save_arbiter_report` and `save_metrics_report` in `ai_battle.py`
2. Copy entire function bodies
3. Preserve all HTML generation and formatting logic
4. Ensure all visualization code is included

**Verification:**
- [ ] Module imports without errors
- [ ] Can generate arbiter report with test data
- [ ] Can generate metrics report with test data
- [ ] HTML output is valid and renders correctly

#### 2.3: Write Unit Tests
**Files:**
- `tests/test_conversation_io.py`
- `tests/test_report_generator.py`

**Required Tests for conversation_io:**
```python
def test_sanitize_filename_part():
    """Test filename sanitization."""
    pass

def test_save_conversation_basic():
    """Test basic conversation saving."""
    pass

def test_save_conversation_with_file_data():
    """Test saving conversation with embedded media."""
    pass

def test_signal_dashboard_rendering():
    """Test signal dashboard HTML generation."""
    pass
```

**Required Tests for report_generator:**
```python
def test_save_arbiter_report():
    """Test arbiter report generation."""
    pass

def test_save_metrics_report():
    """Test metrics report generation."""
    pass
```

**Verification:**
- [ ] All tests pass: `pytest tests/test_conversation_io.py tests/test_report_generator.py -v`
- [ ] Test coverage > 80%

#### 2.4: Update ai_battle.py
**Action:** Replace function definitions with imports

**Find and replace in `ai_battle.py`:**
```python
# OLD: Function definitions in ai_battle.py
async def save_conversation(...):
    # 200+ lines of implementation
    pass

# NEW: Import from io module
from io.conversation_io import save_conversation, _sanitize_filename_part, _render_signal_dashboard
from io.report_generator import save_arbiter_report, save_metrics_report
```

**Verification:**
- [ ] `ai_battle.py` still works
- [ ] Running `python src/ai_battle.py` produces same output
- [ ] Generated HTML files are identical

**Commit Point:**
```bash
git add src/io/ tests/test_conversation_io.py tests/test_report_generator.py src/ai_battle.py
git commit -m "feat: Extract I/O functions to separate modules

- Create io/conversation_io.py for conversation saving
- Create io/report_generator.py for report generation
- Add comprehensive unit tests
- Update ai_battle.py to import from new modules

Part of bootstrap refactor (Phase 2)"
```

---

## Phase 3: Refactor ConversationManager

**Duration:** 4-5 hours
**Goal:** Split ConversationManager into focused components

### Tasks

#### 3.1: Create Client Factory
**File:** `src/core/client_factory.py`

**Action:** Extract client creation and caching logic from ConversationManager

**Template:**
```python
"""Factory for creating and caching model clients."""
import os
import logging
from typing import Dict, Optional
from clients.model_clients import (
    BaseClient,
    OpenAIClient,
    ClaudeClient,
    GeminiClient,
    MLXClient,
    OllamaClient,
)
from lmstudio_client import LMStudioClient
from core.model_registry import get_registry

logger = logging.getLogger(__name__)


class ClientFactory:
    """Factory for creating and caching model clients."""

    def __init__(self):
        """Initialize factory with empty cache."""
        self._cache: Dict[str, BaseClient] = {}
        self._registry = get_registry()

    def get_client(
        self,
        model_name: str,
        config: Optional[dict] = None
    ) -> BaseClient:
        """
        Get or create a client for the specified model.

        Args:
            model_name: Name of the model (e.g., "gpt-4.1", "claude-sonnet-4")
            config: Optional model configuration

        Returns:
            Client instance (cached if previously created)
        """
        # Check cache first
        if model_name in self._cache:
            logger.debug(f"Using cached client for: {model_name}")
            return self._cache[model_name]

        logger.info(f"Creating new client for: {model_name}")

        # Create new client
        client = self._create_client(model_name, config)

        # Cache and return
        self._cache[model_name] = client
        return client

    def _create_client(self, model_name: str, config: Optional[dict]) -> BaseClient:
        """
        Create a new client instance based on model name.

        Args:
            model_name: Name of the model
            config: Optional model configuration

        Returns:
            New client instance

        Raises:
            ValueError: If model provider is unknown
        """
        # TODO: Copy client creation logic from ConversationManager.__init__

        # Determine provider from model name prefix
        if model_name.startswith("gpt-") or model_name.startswith("o") or model_name.startswith("chatgpt"):
            # OpenAI models
            # Get model config from registry
            model_config = self._registry.get_model_config(model_name)
            return OpenAIClient(model_name, model_config or config)

        elif model_name.startswith("claude"):
            # Claude models
            model_config = self._registry.get_model_config(model_name)
            return ClaudeClient(model_name, model_config or config)

        elif model_name.startswith("gemini"):
            # Gemini models
            model_config = self._registry.get_model_config(model_name)
            return GeminiClient(model_name, model_config or config)

        elif model_name.startswith("ollama-"):
            # Ollama models
            return OllamaClient(model_name, config)

        elif model_name.startswith("lmstudio-"):
            # LMStudio models
            return LMStudioClient(model_name, config)

        elif model_name.startswith("mlx-"):
            # MLX models
            return MLXClient(model_name, config)

        else:
            raise ValueError(f"Unknown model provider for: {model_name}")

    def clear_cache(self) -> None:
        """Clear the client cache."""
        logger.info(f"Clearing client cache ({len(self._cache)} clients)")
        self._cache.clear()

    def get_cached_models(self) -> list:
        """Get list of models with cached clients."""
        return list(self._cache.keys())
```

**Implementation Notes:**
1. Find client creation logic in `ConversationManager.__init__` in `ai_battle.py`
2. Extract the logic that determines which client to create
3. Move to `_create_client()` method
4. Add caching logic to avoid recreating clients

**Verification:**
- [ ] Can create OpenAI client: `factory.get_client("gpt-4.1")`
- [ ] Can create Claude client: `factory.get_client("claude-sonnet-4")`
- [ ] Can create Gemini client: `factory.get_client("gemini-2.0-pro")`
- [ ] Caching works: second call returns same instance

#### 3.2: Create Model Discovery Module
**File:** `src/core/model_discovery.py`

**Action:** Extract Ollama and LMStudio model discovery logic

**Template:**
```python
"""Discovery of locally available models (Ollama, LMStudio)."""
import asyncio
import logging
from typing import List, Dict, Any
import aiohttp

logger = logging.getLogger(__name__)


class ModelDiscovery:
    """Discovers locally available models."""

    @staticmethod
    async def discover_ollama_models() -> List[str]:
        """
        Discover available Ollama models via API.

        Returns:
            List of available model names (e.g., ["gemma3:4b-it-q8_0", ...])
        """
        # TODO: Copy from ConversationManager's Ollama discovery logic
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m["name"] for m in data.get("models", [])]
                        logger.info(f"Discovered {len(models)} Ollama models")
                        return models
                    else:
                        logger.warning(f"Ollama API returned status {resp.status}")
                        return []
        except Exception as e:
            logger.debug(f"Could not discover Ollama models: {e}")
            return []

    @staticmethod
    async def discover_lmstudio_models() -> List[str]:
        """
        Discover available LMStudio models via API.

        Returns:
            List of available model names
        """
        # TODO: Copy from ConversationManager's LMStudio discovery logic
        try:
            # LMStudio uses OpenAI-compatible API
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:1234/v1/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m["id"] for m in data.get("data", [])]
                        logger.info(f"Discovered {len(models)} LMStudio models")
                        return models
                    else:
                        logger.warning(f"LMStudio API returned status {resp.status}")
                        return []
        except Exception as e:
            logger.debug(f"Could not discover LMStudio models: {e}")
            return []

    @staticmethod
    async def get_all_local_models() -> Dict[str, List[str]]:
        """
        Get all locally available models from all providers.

        Returns:
            Dictionary with provider names as keys and model lists as values
            Example: {"ollama": ["gemma3:4b", ...], "lmstudio": [...]}
        """
        ollama_models, lmstudio_models = await asyncio.gather(
            ModelDiscovery.discover_ollama_models(),
            ModelDiscovery.discover_lmstudio_models(),
        )

        return {
            "ollama": ollama_models,
            "lmstudio": lmstudio_models,
        }
```

**Implementation Notes:**
1. Find Ollama model discovery in `ConversationManager` (look for "api/tags")
2. Find LMStudio model discovery (look for "v1/models")
3. Extract to static methods
4. Add proper error handling

**Verification:**
- [ ] Can discover Ollama models (if Ollama running)
- [ ] Can discover LMStudio models (if LMStudio running)
- [ ] Returns empty list if service not available (doesn't crash)

#### 3.3: Create Refactored ConversationManager
**File:** `src/core/conversation_manager.py`

**Action:** Create new ConversationManager using ClientFactory and ModelDiscovery

**Template:**
```python
"""Core conversation orchestration."""
import logging
from typing import List, Dict, Any, Optional
from config.configuration import load_config
from config.configdataclasses import DiscussionConfig, FileConfig
from core.client_factory import ClientFactory
from core.model_discovery import ModelDiscovery
from handlers.file_handler import ConversationMediaHandler
from shared_resources import MemoryManager

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Orchestrates conversations between AI models.

    This is the core library class for managing multi-model conversations.
    """

    def __init__(self, config: DiscussionConfig):
        """
        Initialize manager with configuration.

        Args:
            config: Discussion configuration containing models, turns, etc.
        """
        self.config = config
        self.client_factory = ClientFactory()
        self.media_handler = ConversationMediaHandler()
        self.memory_manager = MemoryManager()
        self.signal_history = []

        logger.info(f"Initialized ConversationManager with {len(config.models)} models")

    @classmethod
    def from_config(cls, config_path: str) -> "ConversationManager":
        """
        Create manager from YAML configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ConversationManager instance

        Example:
            >>> manager = ConversationManager.from_config("config.yaml")
        """
        config = load_config(config_path)
        return cls(config)

    def run_conversation(
        self,
        initial_prompt: str,
        human_model: str,
        ai_model: str,
        mode: str = "human-ai",
        rounds: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Run a standard conversation between two models.

        Args:
            initial_prompt: Starting prompt for the conversation
            human_model: Model name for human role
            ai_model: Model name for AI role
            mode: Conversation mode ("human-ai", "ai-ai", "no-meta")
            rounds: Number of conversation rounds

        Returns:
            List of conversation messages in format:
            [
                {"role": "user", "content": "...", "model": "..."},
                {"role": "assistant", "content": "...", "model": "..."},
                ...
            ]

        Example:
            >>> conversation = manager.run_conversation(
            ...     initial_prompt="Discuss AI ethics",
            ...     human_model="claude-sonnet-4",
            ...     ai_model="gpt-4.1",
            ...     rounds=4
            ... )
        """
        # TODO: Copy core conversation orchestration logic from ai_battle.py ConversationManager
        # This is the main conversation loop logic

        logger.info(
            f"Starting conversation: {human_model} <-> {ai_model} "
            f"({mode} mode, {rounds} rounds)"
        )

        # Get clients
        human_client = self.client_factory.get_client(human_model)
        ai_client = self.client_factory.get_client(ai_model)

        conversation = []

        # Implementation goes here...
        # Copy from ai_battle.py ConversationManager.run_conversation()

        logger.info(f"Conversation completed with {len(conversation)} messages")
        return conversation

    def run_conversation_with_file(
        self,
        initial_prompt: str,
        human_model: str,
        ai_model: str,
        file_config: FileConfig,
        mode: str = "ai-ai",
        rounds: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Run a file-based conversation (vision, document analysis, etc.).

        Args:
            initial_prompt: Starting prompt for the conversation
            human_model: Model name for human role
            ai_model: Model name for AI role
            file_config: File configuration (image, video, document)
            mode: Conversation mode
            rounds: Number of conversation rounds

        Returns:
            List of conversation messages with embedded file references

        Example:
            >>> file_config = FileConfig(path="image.jpg", type="image")
            >>> conversation = manager.run_conversation_with_file(
            ...     initial_prompt="Analyze this image",
            ...     human_model="gpt-4.1",
            ...     ai_model="gemini-2.0-pro",
            ...     file_config=file_config
            ... )
        """
        # TODO: Copy file conversation logic from ai_battle.py

        logger.info(
            f"Starting file-based conversation: {human_model} <-> {ai_model} "
            f"(file: {file_config.path})"
        )

        # Process file
        file_metadata = self.media_handler.process_file(file_config.path)

        # Get clients
        human_client = self.client_factory.get_client(human_model)
        ai_client = self.client_factory.get_client(ai_model)

        conversation = []

        # Implementation goes here...
        # Copy from ai_battle.py ConversationManager.run_conversation_with_file()

        logger.info(f"File conversation completed with {len(conversation)} messages")
        return conversation

    # TODO: Add other methods as needed:
    # - _apply_meta_prompting()
    # - _handle_rate_limiting()
    # - _update_signal_history()
    # etc.
```

**Implementation Notes:**
1. This is the largest refactoring task
2. Copy core conversation logic from original ConversationManager
3. Use ClientFactory instead of creating clients directly
4. Preserve ALL functionality - this should work identically to original
5. Keep it focused on orchestration (delegate to factory and discovery)

**Verification:**
- [ ] Can run standard conversation
- [ ] Can run file-based conversation
- [ ] Signal history updates correctly
- [ ] Memory management works
- [ ] Rate limiting works

#### 3.4: Write Tests
**Files:**
- `tests/test_client_factory.py`
- `tests/test_model_discovery.py`
- `tests/test_conversation_manager.py`

**Required Tests:**
```python
# test_client_factory.py
def test_create_openai_client():
    pass

def test_create_claude_client():
    pass

def test_client_caching():
    pass

# test_model_discovery.py
@pytest.mark.asyncio
async def test_discover_ollama_models():
    pass

@pytest.mark.asyncio
async def test_discover_lmstudio_models():
    pass

# test_conversation_manager.py
def test_from_config():
    pass

@pytest.mark.integration
def test_run_conversation():
    """Integration test for standard conversation."""
    pass

@pytest.mark.integration
def test_run_conversation_with_file():
    """Integration test for file-based conversation."""
    pass
```

**Verification:**
- [ ] All unit tests pass
- [ ] Integration tests pass (may require API keys)
- [ ] Test coverage > 70%

#### 3.5: Update ai_battle.py
**Action:** Replace ConversationManager class with import

**In `ai_battle.py`:**
```python
# OLD: Class definition (500+ lines)
class ConversationManager:
    # ... 500 lines ...

# NEW: Import from core module
from core.conversation_manager import ConversationManager
```

**Verification:**
- [ ] `ai_battle.py` still works
- [ ] Running `python src/ai_battle.py` produces same output
- [ ] UI still works: `streamlit run src/ui.py`

**Commit Point:**
```bash
git add src/core/ tests/test_client_factory.py tests/test_model_discovery.py tests/test_conversation_manager.py src/ai_battle.py
git commit -m "feat: Refactor ConversationManager into modular components

- Create ClientFactory for client creation and caching
- Create ModelDiscovery for local model discovery
- Refactor ConversationManager to use factory pattern
- Add comprehensive tests
- Update ai_battle.py to import from new structure

Part of bootstrap refactor (Phase 3)"
```

---

## Phase 4: Create Entry Point Scripts

**Duration:** 3-4 hours
**Goal:** Create focused, configuration-driven entry points

### Tasks

#### 4.1: Create Standard Conversation Runner
**File:** `src/scripts/run_standard.py`

**Complete Implementation:**
```python
#!/usr/bin/env python3
"""
Standard conversation runner - modern replacement for ai_battle.py.

This script runs a single conversation between two models using
configuration files. It's the simplest entry point for the framework.

Usage:
    # Using default config
    python -m scripts.run_standard

    # Using custom config
    python -m scripts.run_standard configs/my_discussion.yaml

    # Override options
    python -m scripts.run_standard configs/discussion.yaml --rounds 8 --output my_conv.html

Examples:
    # Quick 2-round discussion
    python -m scripts.run_standard examples/configs/quick_discussion.yaml

    # Deep 8-round analysis
    python -m scripts.run_standard examples/configs/deep_discussion.yaml --rounds 8
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.conversation_manager import ConversationManager
from io.conversation_io import save_conversation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run a standard conversation from configuration."""
    parser = argparse.ArgumentParser(
        description="Run standard AI conversation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="examples/configs/discussion_config.yaml",
        help="Path to configuration file (default: examples/configs/discussion_config.yaml)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Override number of conversation rounds from config",
    )
    parser.add_argument(
        "--output",
        help="Output filename (default: conversation_<config_name>.html)",
    )
    parser.add_argument(
        "--mode",
        choices=["ai-ai", "human-ai", "no-meta"],
        default="ai-ai",
        help="Conversation mode (default: ai-ai)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 80)
    logger.info("Standard Conversation Runner")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Mode: {args.mode}")

    # Load configuration
    try:
        logger.info(f"Loading configuration from: {args.config}")
        manager = ConversationManager.from_config(args.config)
        logger.info("Configuration loaded successfully")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        return 1
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1

    # Extract model information
    config = manager.config
    models = list(config.models.items())

    if len(models) < 2:
        logger.error("At least two models must be configured")
        return 1

    human_model_id, human_model_config = models[0]
    ai_model_id, ai_model_config = models[1]

    human_model = human_model_config.type
    ai_model = ai_model_config.type
    rounds = args.rounds or config.turns

    logger.info(f"Human model: {human_model} (Role: {human_model_config.role})")
    logger.info(f"AI model: {ai_model} (Role: {ai_model_config.role})")
    logger.info(f"Rounds: {rounds}")
    logger.info(f"Goal: {config.goal[:100]}...")

    # Run conversation
    try:
        logger.info("Starting conversation...")
        conversation = manager.run_conversation(
            initial_prompt=config.goal,
            human_model=human_model,
            ai_model=ai_model,
            mode=args.mode,
            rounds=rounds,
        )
        logger.info(f"Conversation completed with {len(conversation)} messages")
    except Exception as e:
        logger.exception(f"Error during conversation: {e}")
        return 1

    # Save results
    output_file = args.output or f"conversation_{Path(args.config).stem}.html"

    try:
        logger.info(f"Saving conversation to: {output_file}")
        saved_path = await save_conversation(
            conversation=conversation,
            filename=output_file,
            human_model=human_model,
            ai_model=ai_model,
            mode=args.mode,
            signal_history=manager.signal_history,
        )
        logger.info(f"✓ Conversation saved successfully: {saved_path}")
    except Exception as e:
        logger.exception(f"Error saving conversation: {e}")
        return 1

    logger.info("=" * 80)
    logger.info("SUCCESS: Conversation completed and saved")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

**Verification:**
- [ ] Can run with default config: `python -m scripts.run_standard`
- [ ] Can override rounds: `python -m scripts.run_standard --rounds 2`
- [ ] Help works: `python -m scripts.run_standard --help`
- [ ] Generates valid HTML output

#### 4.2: Create Multi-Mode Comparison Runner
**File:** `src/scripts/run_comparison.py`

**Complete Implementation:**
```python
#!/usr/bin/env python3
"""
Multi-mode comparison runner - replicates ai_battle.py comparison functionality.

This script runs conversations in multiple modes and generates comparative
analysis using the arbiter and metrics analyzer.

Usage:
    # Run all three modes (ai-ai, human-ai, no-meta)
    python -m scripts.run_comparison configs/discussion.yaml

    # Run specific modes
    python -m scripts.run_comparison configs/discussion.yaml --modes ai-ai human-ai

    # Specify output directory
    python -m scripts.run_comparison configs/discussion.yaml --output-dir ./results

Examples:
    # Full comparison with all modes
    python -m scripts.run_comparison examples/configs/discussion_config.yaml

    # Quick comparison (2 modes only)
    python -m scripts.run_comparison examples/configs/quick_discussion.yaml --modes ai-ai no-meta
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.conversation_manager import ConversationManager
from io.conversation_io import save_conversation
from io.report_generator import save_arbiter_report, save_metrics_report
from analysis.arbiter_v4 import evaluate_conversations
from analysis.metrics_analyzer import analyze_conversations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run multi-mode conversation comparison."""
    parser = argparse.ArgumentParser(
        description="Run multi-mode conversation comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="examples/configs/discussion_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["ai-ai", "human-ai", "no-meta"],
        choices=["ai-ai", "human-ai", "no-meta"],
        help="Conversation modes to run (default: all three)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for results (default: ./output)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Override number of rounds from config",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Multi-Mode Conversation Comparison")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Modes: {', '.join(args.modes)}")
    logger.info(f"Output directory: {output_dir}")

    # Load configuration
    try:
        logger.info("Loading configuration...")
        manager = ConversationManager.from_config(args.config)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1

    # Extract model information
    config = manager.config
    models = list(config.models.items())

    if len(models) < 2:
        logger.error("At least two models must be configured")
        return 1

    human_model = models[0][1].type
    ai_model = models[1][1].type
    rounds = args.rounds or config.turns

    logger.info(f"Models: {human_model} <-> {ai_model}")
    logger.info(f"Rounds: {rounds}")

    conversations = {}

    # Run conversations in each mode
    for i, mode in enumerate(args.modes, 1):
        logger.info("=" * 80)
        logger.info(f"Running conversation {i}/{len(args.modes)}: {mode} mode")
        logger.info("=" * 80)

        try:
            conversation = manager.run_conversation(
                initial_prompt=config.goal,
                human_model=human_model,
                ai_model=ai_model,
                mode=mode,
                rounds=rounds,
            )

            conversations[mode] = conversation
            logger.info(f"✓ {mode} conversation completed ({len(conversation)} messages)")

            # Save individual conversation
            output_file = str(output_dir / f"conversation_{mode}.html")
            await save_conversation(
                conversation=conversation,
                filename=output_file,
                human_model=human_model,
                ai_model=ai_model,
                mode=mode,
                signal_history=manager.signal_history,
            )
            logger.info(f"✓ Saved to: {output_file}")

        except Exception as e:
            logger.exception(f"Error in {mode} mode: {e}")
            return 1

    # Run arbiter evaluation
    logger.info("=" * 80)
    logger.info("Running arbiter evaluation...")
    logger.info("=" * 80)

    try:
        evaluation = await evaluate_conversations(
            conversations=list(conversations.values()),
            modes=args.modes,
        )

        arbiter_file = str(output_dir / "arbiter_report.html")
        await save_arbiter_report(
            evaluation_result=evaluation,
            filename=arbiter_file,
            human_model=human_model,
            ai_model=ai_model,
        )
        logger.info(f"✓ Arbiter report saved: {arbiter_file}")

    except Exception as e:
        logger.exception(f"Error running arbiter: {e}")
        # Don't fail the whole run if arbiter fails
        logger.warning("Continuing without arbiter report...")

    # Run metrics analysis
    logger.info("=" * 80)
    logger.info("Running metrics analysis...")
    logger.info("=" * 80)

    try:
        metrics = await analyze_conversations(
            conversations=list(conversations.values()),
        )

        metrics_file = str(output_dir / "metrics_report.html")
        await save_metrics_report(
            metrics_result=metrics,
            filename=metrics_file,
        )
        logger.info(f"✓ Metrics report saved: {metrics_file}")

    except Exception as e:
        logger.exception(f"Error running metrics: {e}")
        # Don't fail the whole run if metrics fails
        logger.warning("Continuing without metrics report...")

    logger.info("=" * 80)
    logger.info("SUCCESS: Multi-mode comparison completed!")
    logger.info("=" * 80)
    logger.info(f"Results saved in: {output_dir}")
    logger.info(f"- {len(args.modes)} conversation HTML files")
    logger.info(f"- arbiter_report.html (if successful)")
    logger.info(f"- metrics_report.html (if successful)")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

**Verification:**
- [ ] Can run all modes: `python -m scripts.run_comparison`
- [ ] Can specify modes: `python -m scripts.run_comparison --modes ai-ai human-ai`
- [ ] Creates output directory
- [ ] Generates all expected files

#### 4.3: Move and Fix Vision Script
**File:** `src/scripts/run_vision.py`

**Action:**
1. Move `docs/examples/run_vision_discussion.py` to `src/scripts/run_vision.py`
2. Fix typo on line 29: `sys.path.apparent(scr_dir)` → `sys.path.append(src_dir)`
3. Update imports to use new structure
4. Add proper command-line help

**Commands:**
```bash
# Move file
mv docs/examples/run_vision_discussion.py src/scripts/run_vision.py

# Fix in src/scripts/run_vision.py
```

**Updates needed in run_vision.py:**
```python
# Line 29: Fix typo
sys.path.append(src_dir)  # Was: sys.path.apparent(scr_dir)

# Lines 31-38: Remove importlib hack, use direct import
# OLD:
# ai_battle_path = os.path.join(src_dir, "ai-battle.py")
# spec = importlib.util.spec_from_file_location("ai_battle", ai_battle_path)
# ai_battle = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(ai_battle)
# ConversationManager = ai_battle.ConversationManager
# save_conversation = ai_battle.save_conversation

# NEW:
from core.conversation_manager import ConversationManager
from io.conversation_io import save_conversation
```

**Verification:**
- [ ] Script runs without import errors
- [ ] Can run vision discussion: `python -m scripts.run_vision configs/vision_discussion.yaml`
- [ ] Can process images and videos
- [ ] Generates valid HTML with embedded media

#### 4.4: Write Integration Tests
**File:** `tests/integration/test_entry_points.py`

**Required Tests:**
```python
@pytest.mark.integration
def test_run_standard_script():
    """Test running standard conversation script."""
    pass

@pytest.mark.integration
def test_run_comparison_script():
    """Test running comparison script."""
    pass

@pytest.mark.integration
def test_run_vision_script():
    """Test running vision script."""
    pass
```

**Verification:**
- [ ] Integration tests pass
- [ ] Scripts can be run programmatically
- [ ] Output files are generated correctly

#### 4.5: Update Documentation
**Files:**
- `README.md`
- `docs/usage.md` (create if doesn't exist)

**Add to README.md:**
```markdown
## Usage

### Quick Start

Run a standard conversation:
\`\`\`bash
python -m scripts.run_standard examples/configs/discussion_config.yaml
\`\`\`

Run a multi-mode comparison:
\`\`\`bash
python -m scripts.run_comparison examples/configs/discussion_config.yaml
\`\`\`

Run a vision-based discussion:
\`\`\`bash
python -m scripts.run_vision examples/configs/vision_discussion.yaml
\`\`\`

### Entry Points

- **run_standard.py**: Single conversation with configuration
- **run_comparison.py**: Multi-mode comparison with analysis
- **run_vision.py**: Vision-based conversations (images, videos)

See `python -m scripts.<script_name> --help` for options.

### Configuration

All entry points use YAML configuration files. See `examples/configs/` for examples.

Example configuration:
\`\`\`yaml
discussion:
  turns: 4
  models:
    model1:
      type: "claude-sonnet-4"
      role: "human"
      persona: |
        You are a thoughtful analyst...
    model2:
      type: "gpt-4.1"
      role: "assistant"
      persona: |
        You are a helpful assistant...
  goal: |
    Discuss the implications of...
\`\`\`
```

**Verification:**
- [ ] Documentation is clear and accurate
- [ ] Examples work as documented
- [ ] Links are valid

**Commit Point:**
```bash
git add src/scripts/ tests/integration/ README.md docs/
git commit -m "feat: Create configuration-driven entry point scripts

- Add run_standard.py for single conversations
- Add run_comparison.py for multi-mode analysis
- Move and fix run_vision.py
- Add comprehensive command-line interfaces
- Add integration tests
- Update documentation

Part of bootstrap refactor (Phase 4)"
```

---

## Phase 5: Deprecate ai_battle.py

**Duration:** 1-2 hours
**Goal:** Complete migration and remove monolithic file

### Tasks

#### 5.1: Create Deprecation Wrapper
**File:** `src/ai_battle.py` (replace contents)

**Implementation:**
```python
"""
DEPRECATED: This file is deprecated and will be removed in a future release.

The monolithic ai_battle.py has been refactored into a modular structure.

NEW STRUCTURE:
--------------
Entry Points:
- scripts.run_standard: Standard conversations
- scripts.run_comparison: Multi-mode comparisons
- scripts.run_vision: Vision-based discussions

Library Modules:
- core.conversation_manager: ConversationManager
- core.model_registry: Model configurations
- core.client_factory: Client creation and caching
- core.model_discovery: Local model discovery
- io.conversation_io: Conversation saving
- io.report_generator: Report generation

MIGRATION GUIDE:
---------------

1. Replace script execution:
   OLD: python src/ai_battle.py
   NEW: python -m scripts.run_comparison

2. Update imports:
   OLD: from ai_battle import ConversationManager, save_conversation
   NEW: from core.conversation_manager import ConversationManager
        from io.conversation_io import save_conversation

3. Update your code:
   The API is identical, only import paths have changed.

For detailed migration instructions, see MIGRATION_GUIDE.md
"""
import warnings
import sys

# Issue deprecation warning
warnings.warn(
    "ai_battle.py is deprecated. Use scripts.run_standard, scripts.run_comparison, "
    "or scripts.run_vision instead. "
    "See MIGRATION_GUIDE.md for details.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backward compatibility
# This allows existing code to continue working during migration
try:
    from core.conversation_manager import ConversationManager
    from core.model_registry import get_registry, ModelRegistry
    from core.client_factory import ClientFactory
    from core.model_discovery import ModelDiscovery
    from io.conversation_io import save_conversation
    from io.report_generator import save_arbiter_report, save_metrics_report

    # Export model registries for backward compatibility
    _registry = get_registry()
    OPENAI_MODELS = _registry.openai_models
    GEMINI_MODELS = _registry.gemini_models
    CLAUDE_MODELS = _registry.claude_models
    OLLAMA_THINKING_CONFIG = _registry.ollama_thinking_config

    __all__ = [
        "ConversationManager",
        "ModelRegistry",
        "ClientFactory",
        "ModelDiscovery",
        "save_conversation",
        "save_arbiter_report",
        "save_metrics_report",
        "get_registry",
        "OPENAI_MODELS",
        "GEMINI_MODELS",
        "CLAUDE_MODELS",
        "OLLAMA_THINKING_CONFIG",
    ]

except ImportError as e:
    print(f"ERROR: Could not import refactored modules: {e}", file=sys.stderr)
    print("The refactor may be incomplete. Check the implementation.", file=sys.stderr)
    sys.exit(1)

# If someone tries to run this file directly, show error and redirect
if __name__ == "__main__":
    print("=" * 80)
    print("ERROR: ai_battle.py is deprecated and can no longer be run directly")
    print("=" * 80)
    print()
    print("Please use one of the new entry points instead:")
    print()
    print("  1. Standard conversation:")
    print("     python -m scripts.run_standard [config.yaml]")
    print()
    print("  2. Multi-mode comparison:")
    print("     python -m scripts.run_comparison [config.yaml]")
    print()
    print("  3. Vision-based discussion:")
    print("     python -m scripts.run_vision [config.yaml] [files...]")
    print()
    print("For more information:")
    print("  python -m scripts.run_standard --help")
    print()
    print("See MIGRATION_GUIDE.md for detailed migration instructions.")
    print("=" * 80)
    sys.exit(1)
```

**Verification:**
- [ ] Warning is shown when importing from ai_battle
- [ ] Re-exports work: `from ai_battle import ConversationManager`
- [ ] Running directly shows helpful error message
- [ ] Error message includes correct migration paths

#### 5.2: Update All Imports
**Action:** Find and update all imports across the codebase

**Find imports:**
```bash
# Find all imports from ai_battle
grep -r "from ai_battle import" src/ --exclude-dir=__pycache__
grep -r "import ai_battle" src/ --exclude-dir=__pycache__
```

**Update:**
- `src/ui.py`: Update imports
- Any other files importing from `ai_battle`

**Example:**
```python
# OLD
from ai_battle import ConversationManager, save_conversation

# NEW
from core.conversation_manager import ConversationManager
from io.conversation_io import save_conversation
```

**Verification:**
- [ ] No imports from `ai_battle` except tests for backward compatibility
- [ ] All files import from new structure
- [ ] No import errors

#### 5.3: Update Streamlit UI
**File:** `src/ui.py`

**Action:** Update imports to use new structure

**Find and replace:**
```python
# OLD imports
from ai_battle import ConversationManager, save_conversation

# NEW imports
from core.conversation_manager import ConversationManager
from io.conversation_io import save_conversation
from core.model_discovery import ModelDiscovery
```

**Verification:**
- [ ] UI launches: `streamlit run src/ui.py`
- [ ] Can run conversations through UI
- [ ] No deprecation warnings in UI usage
- [ ] All features work identically

#### 5.4: Run Full Test Suite
**Action:** Verify everything works together

**Commands:**
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run integration tests
pytest tests/integration/ -v -m integration

# Test all entry points
python -m scripts.run_standard examples/configs/discussion_config.yaml --rounds 1
python -m scripts.run_comparison examples/configs/discussion_config.yaml --modes ai-ai --rounds 1
python -m scripts.run_vision examples/configs/vision_discussion.yaml

# Test UI (manual verification)
streamlit run src/ui.py
```

**Verification Checklist:**
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Test coverage > 70%
- [ ] run_standard works
- [ ] run_comparison works
- [ ] run_vision works
- [ ] UI works
- [ ] No import errors
- [ ] No functionality regressions

#### 5.5: Create Migration Guide
**File:** `MIGRATION_GUIDE.md`

**Content:**
```markdown
# Migration Guide: ai_battle.py → Modular Structure

This guide helps you migrate from the deprecated `ai_battle.py` to the new modular structure.

## For Script Users

### Running Conversations

**OLD:**
\`\`\`bash
python src/ai_battle.py
\`\`\`

**NEW:**
\`\`\`bash
# Standard conversation
python -m scripts.run_standard examples/configs/discussion_config.yaml

# Multi-mode comparison (replaces old ai_battle.py behavior)
python -m scripts.run_comparison examples/configs/discussion_config.yaml

# Vision-based conversation
python -m scripts.run_vision examples/configs/vision_discussion.yaml
\`\`\`

## For Library Users

### Importing Classes

**OLD:**
\`\`\`python
from ai_battle import (
    ConversationManager,
    save_conversation,
    save_arbiter_report,
    OPENAI_MODELS,
)
\`\`\`

**NEW:**
\`\`\`python
from core.conversation_manager import ConversationManager
from io.conversation_io import save_conversation
from io.report_generator import save_arbiter_report
from core.model_registry import get_registry

# Access model registries
registry = get_registry()
openai_models = registry.openai_models
\`\`\`

### Using ConversationManager

The API is **identical**, only import paths changed:

\`\`\`python
# Same as before, just different import
from core.conversation_manager import ConversationManager

manager = ConversationManager.from_config("config.yaml")
conversation = manager.run_conversation(
    initial_prompt="Discuss AI ethics",
    human_model="claude-sonnet-4",
    ai_model="gpt-4.1",
    rounds=4,
)
\`\`\`

## Backward Compatibility

For now, `ai_battle.py` re-exports everything for backward compatibility:

\`\`\`python
# This still works but shows a deprecation warning
from ai_battle import ConversationManager  # ⚠️ Deprecated
\`\`\`

**Action Required:** Update your imports to the new structure before the next major release.

## New Features

The refactor adds new capabilities:

1. **Configuration-driven entry points** with command-line interfaces
2. **Model registry** in YAML for easy model management
3. **Focused modules** for better code organization
4. **Better testability** with separated concerns

## Timeline

- **Current:** ai_battle.py shows deprecation warnings but still works
- **v2.0:** ai_battle.py will be removed entirely

Please migrate your code before v2.0.

## Need Help?

Open an issue if you encounter migration problems: [GitHub Issues](https://github.com/your-repo/issues)
```

**Verification:**
- [ ] Migration guide is clear
- [ ] Examples are accurate
- [ ] Timeline is specified

#### 5.6: Update CI/CD
**File:** `.github/workflows/test.yml` (if exists)

**Action:** Update CI/CD pipelines to:
1. Test new entry points
2. Verify backward compatibility
3. Check for deprecation warnings

**Verification:**
- [ ] CI tests pass
- [ ] All entry points tested
- [ ] No unexpected warnings

**Commit Point:**
```bash
git add src/ai_battle.py src/ui.py MIGRATION_GUIDE.md .github/
git commit -m "feat: Add deprecation wrapper and migration guide

- Replace ai_battle.py with deprecation wrapper
- Maintain backward compatibility via re-exports
- Update UI to use new structure
- Create comprehensive migration guide
- Update CI/CD pipelines

Part of bootstrap refactor (Phase 5)"
```

---

## Phase 6: Final Verification and Cleanup

**Duration:** 2-3 hours
**Goal:** Ensure everything works, document, and clean up

### Tasks

#### 6.1: Comprehensive Testing

**Run full test matrix:**
```bash
# 1. Unit tests
pytest tests/ -v --cov=src --cov-report=term-missing

# 2. Integration tests
pytest tests/integration/ -v -m integration

# 3. Test all entry points with various configs
for config in examples/configs/*.yaml; do
    echo "Testing $config..."
    python -m scripts.run_standard "$config" --rounds 1
done

# 4. Test UI
streamlit run src/ui.py &
# Manual verification, then:
kill %1

# 5. Test backward compatibility
python -c "from ai_battle import ConversationManager; print('✓ Backward compat works')"

# 6. Check for deprecation warnings
python -W error::DeprecationWarning -c "from ai_battle import ConversationManager" 2>&1 | grep -q "DeprecationWarning" && echo "✓ Deprecation warning present"
```

**Verification Checklist:**
- [ ] All unit tests pass (> 70% coverage)
- [ ] All integration tests pass
- [ ] All example configs work
- [ ] UI functions correctly
- [ ] Backward compatibility maintained
- [ ] Deprecation warnings shown
- [ ] No import errors
- [ ] No functionality regressions

#### 6.2: Code Quality Checks

**Run linters and formatters:**
```bash
# Format code (if using black)
black src/ tests/

# Lint code (if using pylint/flake8)
pylint src/
flake8 src/

# Type checking (if using mypy)
mypy src/

# Check for security issues (if using bandit)
bandit -r src/
```

**Verification:**
- [ ] No linter errors
- [ ] Code formatted consistently
- [ ] No type errors (if using type hints)
- [ ] No security issues

#### 6.3: Documentation Review

**Update all documentation:**

1. **README.md:**
   - [ ] Updated architecture diagram
   - [ ] Updated usage examples
   - [ ] Links to new entry points
   - [ ] Migration guide link

2. **ARCHITECTURE.md** (create if doesn't exist):
   ```markdown
   # Architecture

   ## Directory Structure

   \`\`\`
   src/
   ├── core/              # Core orchestration
   ├── io/                # I/O operations
   ├── config/            # Configuration
   ├── clients/           # Model clients
   ├── handlers/          # File handlers
   ├── analysis/          # Analysis components
   ├── scripts/           # Entry points
   └── data/              # Data files
   \`\`\`

   ## Component Relationships

   [Diagram or description]

   ## Key Classes

   - ConversationManager: Orchestrates conversations
   - ClientFactory: Creates and caches clients
   - ModelRegistry: Manages model configurations

   [etc.]
   ```

3. **API Documentation:**
   - [ ] Generate API docs (if using Sphinx/pdoc)
   - [ ] Document all public APIs
   - [ ] Include examples

4. **Example Configurations:**
   - [ ] Verify all examples work
   - [ ] Add comments explaining options
   - [ ] Create templates for common use cases

**Verification:**
- [ ] All documentation accurate
- [ ] Examples work as documented
- [ ] API docs generated
- [ ] Links valid

#### 6.4: Performance Testing

**Compare performance:**
```bash
# Benchmark before (if you saved old version)
time python src/ai_battle.py.backup

# Benchmark after
time python -m scripts.run_comparison examples/configs/discussion_config.yaml --rounds 1

# Should be similar performance
```

**Verification:**
- [ ] No significant performance regression
- [ ] Import times reasonable
- [ ] Memory usage similar

#### 6.5: Create Summary Report

**File:** `REFACTOR_SUMMARY.md`

**Content:**
```markdown
# Bootstrap Refactor Summary

## Changes Made

### Phase 1: Model Registry
- Extracted model configurations to `src/data/model_registry.yaml`
- Created `ModelRegistry` class for loading and managing models
- Added 8 unit tests with 95% coverage

### Phase 2: I/O Separation
- Separated conversation I/O to `src/io/conversation_io.py`
- Separated report generation to `src/io/report_generator.py`
- Added 6 unit tests with 85% coverage

### Phase 3: ConversationManager Refactor
- Split into:
  - `ClientFactory` (100 lines)
  - `ModelDiscovery` (150 lines)
  - `ConversationManager` (300 lines, down from 500)
- Added 12 unit tests with 75% coverage
- Added 3 integration tests

### Phase 4: Entry Point Scripts
- Created `run_standard.py` (80 lines)
- Created `run_comparison.py` (120 lines)
- Fixed and moved `run_vision.py`
- Added CLI interfaces for all scripts
- Added 3 integration tests

### Phase 5: Deprecation
- Created deprecation wrapper in `ai_battle.py`
- Updated all imports across codebase
- Created migration guide
- Verified backward compatibility

### Phase 6: Verification
- All tests pass (73% overall coverage)
- All entry points work
- UI works identically
- Documentation updated

## Metrics

### Code Reduction
- **Before:** 2,243 lines in 1 file
- **After:** 1,350 lines across 9 files
- **Reduction:** 40% fewer lines

### File Sizes
- Largest file: 300 lines (conversation_manager.py)
- Average file: 150 lines
- All files < 500 lines ✓

### Test Coverage
- Unit tests: 73%
- Integration tests: 3
- Total tests: 32

### Documentation
- README updated ✓
- Migration guide created ✓
- API docs generated ✓
- Examples updated ✓

## Benefits Achieved

1. **Maintainability:** Each module has single responsibility
2. **Testability:** Components can be tested in isolation
3. **Reusability:** Import only what you need
4. **Flexibility:** Configuration-driven entry points
5. **Clarity:** Logical directory structure

## Backward Compatibility

✅ Maintained via deprecation wrapper
✅ All existing code continues to work
⚠️  Deprecation warnings shown (as intended)

## Next Steps

1. Monitor for issues after deployment
2. Assist users with migration
3. Remove ai_battle.py in v2.0 (after deprecation period)
4. Consider additional optimizations

## Timeline

- Planning: 1 day
- Implementation: 3 weeks (15-20 hours)
- Total: Completed on schedule

## Conclusion

Successfully refactored monolithic bootstrap mechanism into clean, modular architecture while maintaining 100% backward compatibility and functionality.
```

**Verification:**
- [ ] Summary is accurate
- [ ] Metrics are correct
- [ ] Benefits documented

#### 6.6: Final Cleanup

**Remove temporary files:**
```bash
# Remove backup if everything works
rm src/ai_battle.py.backup

# Clean up any test outputs
rm -rf output/ conversation_*.html arbiter_report.html metrics_report.html

# Clean Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

**Verification:**
- [ ] No leftover backup files
- [ ] No test outputs in repo
- [ ] No cache files

**Final Commit:**
```bash
git add -A
git commit -m "docs: Add comprehensive documentation and cleanup

- Add refactor summary report
- Update all documentation
- Generate API docs
- Clean up temporary files

Completes bootstrap refactor"
```

---

## Delivery Checklist

Before marking this assignment complete, verify:

### Functionality
- [ ] All existing functionality preserved
- [ ] All entry points work correctly
- [ ] UI works identically
- [ ] All example configurations work
- [ ] Backward compatibility maintained

### Code Quality
- [ ] No file > 500 lines
- [ ] All modules focused (single responsibility)
- [ ] Test coverage > 70%
- [ ] All linter checks pass
- [ ] No security issues

### Documentation
- [ ] README updated
- [ ] Migration guide created
- [ ] API documentation generated
- [ ] All examples work
- [ ] Architecture documented

### Testing
- [ ] 32+ unit tests (all passing)
- [ ] 3+ integration tests (all passing)
- [ ] Manual testing completed
- [ ] Performance verified
- [ ] UI verified

### Git
- [ ] All changes committed
- [ ] Commit messages clear
- [ ] Branch ready for PR
- [ ] No uncommitted changes

---

## Success Criteria

This assignment is successful when:

1. ✅ All code from `ai_battle.py` moved to appropriate modules
2. ✅ All functionality works identically
3. ✅ Test coverage > 70%
4. ✅ Documentation complete
5. ✅ UI continues to work
6. ✅ Backward compatibility maintained
7. ✅ Code quality improved (no file > 500 lines)
8. ✅ Entry points are configuration-driven

---

## Need Help?

If you encounter issues:

1. **Check the proposal:** `BOOTSTRAP_REFACTOR_PROPOSAL.md` has detailed architecture
2. **Review original code:** `src/ai_battle.py` is the source of truth
3. **Test incrementally:** Commit after each phase
4. **Verify backward compat:** After each phase, ensure `python src/ai_battle.py` still works
5. **Ask questions:** Document any ambiguities encountered

---

## Estimated Time Breakdown

- Phase 1 (Model Registry): 2-3 hours
- Phase 2 (I/O Separation): 2-3 hours
- Phase 3 (ConversationManager): 4-5 hours
- Phase 4 (Entry Points): 3-4 hours
- Phase 5 (Deprecation): 1-2 hours
- Phase 6 (Verification): 2-3 hours

**Total: 15-20 hours** across 3 weeks

---

## Final Notes

This is a **large refactoring** that requires careful attention to:

1. **Preserving functionality:** Every feature must work identically
2. **Maintaining compatibility:** Existing code must continue working
3. **Testing thoroughly:** Test after each phase
4. **Documenting clearly:** Future developers need to understand the new structure

Take your time, test thoroughly, and commit frequently. Good luck! 🚀
