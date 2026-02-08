# Bootstrap Refactor: Agent Task Specification

## The Problem

`src/ai_battle.py` is 2,243 lines that mixes:
- Model config dicts (OPENAI_MODELS, GEMINI_MODELS, CLAUDE_MODELS)
- ConversationManager class (500+ lines)
- I/O functions (save_conversation, save_arbiter_report, etc.)
- Hard-coded bootstrap logic in main()

You can't import ConversationManager without loading all this crap. Configuration is hard-coded. It's a mess.

## The Solution

`docs/examples/run_vision_discussion.py` shows the right pattern:
```python
manager = ConversationManager.from_config(config_path)
conversation = manager.run_conversation_with_file(...)
await save_conversation(...)
```

Configuration-driven, clean separation, single purpose.

## What Needs to Happen

Break apart `ai_battle.py` into:

```
src/
├── core/
│   ├── conversation_manager.py     # ConversationManager (refactored)
│   ├── client_factory.py           # Client creation logic
│   ├── model_registry.py           # Loads model configs from YAML
│   └── model_discovery.py          # Ollama/LMStudio discovery
├── io/
│   ├── conversation_io.py          # save_conversation()
│   └── report_generator.py         # save_arbiter_report(), save_metrics_report()
├── data/
│   └── model_registry.yaml         # All model configs (externalized)
└── scripts/
    ├── run_standard.py             # Single conversation
    ├── run_comparison.py           # Multi-mode (replaces old main())
    └── run_vision.py               # Fix and move from docs/examples/
```

## Task for Agent

### Step 1: Model Registry (Start Here)

**Create `src/data/model_registry.yaml`:**
```yaml
openai_models:
  o1:
    model: "o1"
    reasoning_level: "medium"
    multimodal: false
  # ... copy all from OPENAI_MODELS dict in ai_battle.py

gemini_models:
  # ... copy all from GEMINI_MODELS dict

claude_models:
  # ... copy all from CLAUDE_MODELS dict
```

**Create `src/core/model_registry.py`:**
```python
import yaml
from pathlib import Path

class ModelRegistry:
    def __init__(self, registry_path=None):
        if not registry_path:
            registry_path = Path(__file__).parent.parent / "data" / "model_registry.yaml"

        with open(registry_path) as f:
            data = yaml.safe_load(f)

        self.openai_models = data.get("openai_models", {})
        self.gemini_models = data.get("gemini_models", {})
        self.claude_models = data.get("claude_models", {})

    def get_model_config(self, model_name):
        for registry in [self.openai_models, self.gemini_models, self.claude_models]:
            if model_name in registry:
                return registry[model_name]
        return None

_registry = None
def get_registry():
    global _registry
    if not _registry:
        _registry = ModelRegistry()
    return _registry
```

**Test it works:**
```python
from core.model_registry import get_registry
registry = get_registry()
config = registry.get_model_config("o3")
print(config)  # Should print {'model': 'o3', ...}
```

### Step 2: Extract I/O Functions

**Create `src/io/conversation_io.py`:**
- Copy `save_conversation()` from ai_battle.py
- Copy `_sanitize_filename_part()` helper
- Copy `_render_signal_dashboard()` helper

**Create `src/io/report_generator.py`:**
- Copy `save_arbiter_report()` from ai_battle.py
- Copy `save_metrics_report()` from ai_battle.py

**Test:**
```python
from io.conversation_io import save_conversation
# Should import without errors
```

### Step 3: Client Factory

**Create `src/core/client_factory.py`:**
```python
from clients.model_clients import *
from lmstudio_client import LMStudioClient
from core.model_registry import get_registry

class ClientFactory:
    def __init__(self):
        self._cache = {}
        self._registry = get_registry()

    def get_client(self, model_name, config=None):
        if model_name in self._cache:
            return self._cache[model_name]

        client = self._create_client(model_name, config)
        self._cache[model_name] = client
        return client

    def _create_client(self, model_name, config):
        # Extract client creation logic from ConversationManager.__init__
        if model_name.startswith("gpt-") or model_name.startswith("o"):
            return OpenAIClient(model_name, config)
        elif model_name.startswith("claude"):
            return ClaudeClient(model_name, config)
        elif model_name.startswith("gemini"):
            return GeminiClient(model_name, config)
        elif model_name.startswith("ollama-"):
            return OllamaClient(model_name, config)
        elif model_name.startswith("lmstudio-"):
            return LMStudioClient(model_name, config)
        elif model_name.startswith("mlx-"):
            return MLXClient(model_name, config)
        else:
            raise ValueError(f"Unknown model: {model_name}")
```

### Step 4: Model Discovery

**Create `src/core/model_discovery.py`:**
```python
import aiohttp

class ModelDiscovery:
    @staticmethod
    async def discover_ollama_models():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return [m["name"] for m in data.get("models", [])]
        except:
            return []

    @staticmethod
    async def discover_lmstudio_models():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:1234/v1/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return [m["id"] for m in data.get("data", [])]
        except:
            return []
```

### Step 5: Refactor ConversationManager

**Create `src/core/conversation_manager.py`:**
- Copy ConversationManager class from ai_battle.py
- Replace client creation with `self.client_factory = ClientFactory()`
- Use `self.client_factory.get_client(model_name)` instead of creating directly
- Remove model discovery code (use ModelDiscovery instead)
- Keep all orchestration logic

**Key:**
```python
from core.client_factory import ClientFactory

class ConversationManager:
    def __init__(self, config):
        self.config = config
        self.client_factory = ClientFactory()
        # ... rest of init

    def run_conversation(self, ...):
        human_client = self.client_factory.get_client(human_model)
        ai_client = self.client_factory.get_client(ai_model)
        # ... rest of logic
```

### Step 6: Entry Point Scripts

**Create `src/scripts/run_standard.py`:**
```python
#!/usr/bin/env python3
import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.conversation_manager import ConversationManager
from io.conversation_io import save_conversation

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="examples/configs/discussion_config.yaml")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    manager = ConversationManager.from_config(args.config)
    config = manager.config
    models = list(config.models.items())

    human_model = models[0][1].type
    ai_model = models[1][1].type
    rounds = args.rounds or config.turns

    conversation = manager.run_conversation(
        initial_prompt=config.goal,
        human_model=human_model,
        ai_model=ai_model,
        mode="ai-ai",
        rounds=rounds,
    )

    output_file = args.output or f"conversation_{Path(args.config).stem}.html"
    await save_conversation(
        conversation=conversation,
        filename=output_file,
        human_model=human_model,
        ai_model=ai_model,
        signal_history=manager.signal_history,
    )

    print(f"Saved: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Create `src/scripts/run_comparison.py`:**
Similar to run_standard.py but runs multiple modes and generates reports (replicates old ai_battle.py main() behavior).

**Fix `docs/examples/run_vision_discussion.py`:**
- Line 29: `sys.path.apparent(scr_dir)` → `sys.path.append(src_dir)`
- Replace importlib hack with: `from core.conversation_manager import ConversationManager`
- Move to `src/scripts/run_vision.py`

### Step 7: Update ai_battle.py (Backward Compat)

**Replace `src/ai_battle.py` with:**
```python
"""
DEPRECATED: Use scripts.run_standard, scripts.run_comparison, or scripts.run_vision instead.

Backward compatibility: imports still work but show warnings.
"""
import warnings

warnings.warn(
    "ai_battle.py is deprecated. Use scripts.run_* instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for compatibility
from core.conversation_manager import ConversationManager
from core.model_registry import get_registry
from io.conversation_io import save_conversation
from io.report_generator import save_arbiter_report, save_metrics_report

__all__ = [
    "ConversationManager",
    "save_conversation",
    "save_arbiter_report",
    "save_metrics_report",
    "get_registry",
]

if __name__ == "__main__":
    print("ERROR: ai_battle.py is deprecated.")
    print("Use: python -m scripts.run_comparison [config.yaml]")
    exit(1)
```

### Step 8: Update ui.py

**In `src/ui.py`, replace:**
```python
# OLD
from ai_battle import ConversationManager, save_conversation

# NEW
from core.conversation_manager import ConversationManager
from io.conversation_io import save_conversation
```

Test: `streamlit run src/ui.py` should work identically.

### Step 9: Verify

**Test everything:**
```bash
# Test imports
python -c "from core.conversation_manager import ConversationManager; print('✓')"
python -c "from io.conversation_io import save_conversation; print('✓')"
python -c "from core.model_registry import get_registry; print('✓')"

# Test entry points
python -m scripts.run_standard examples/configs/discussion_config.yaml --rounds 1
python -m scripts.run_comparison examples/configs/discussion_config.yaml --rounds 1

# Test UI
streamlit run src/ui.py

# Test backward compat (should show warning)
python -c "from ai_battle import ConversationManager"
```

## Critical Requirements

1. **Preserve all functionality** - Everything must work identically
2. **Backward compatibility** - Old imports keep working (with warnings)
3. **UI must work** - `streamlit run src/ui.py` unchanged
4. **No hard-coded configs** - Models in YAML, not Python
5. **Clean imports** - No side effects when importing

## What Success Looks Like

After refactor:
```bash
# Run single conversation
python -m scripts.run_standard config.yaml

# Run multi-mode comparison (old behavior)
python -m scripts.run_comparison config.yaml

# Run vision discussion
python -m scripts.run_vision config.yaml image.jpg

# Use as library
python -c "from core.conversation_manager import ConversationManager; ..."
```

All existing code continues working (with deprecation warnings for ai_battle imports).

## Notes for Agent

- Focus on moving code, not rewriting it
- Test after each major move
- Keep commits focused (one per major change)
- If something doesn't work, check imports and paths
- ConversationManager orchestration logic stays mostly the same
- Main changes are: client creation (factory), model configs (YAML), separated I/O

Don't overthink it. Just extract, move, wire up imports, test.
