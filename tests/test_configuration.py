import pytest
from pathlib import Path
import yaml
from configuration import (
    load_config,
    TimeoutConfig,
    detect_model_capabilities,
)
from configdataclasses import DiscussionConfig, ModelConfig, FileConfig


def test_valid_config(tmp_path):
    """Test loading a valid configuration file"""
    config_path = tmp_path / "test_config.yaml"
    config_content = """
discussion:
  turns: 3
  models:
    model1:
      type: "claude-3-sonnet"
      role: "human"
      instructions:
        template: "human_system_instructions"
        params:
          domain: "Quantum Computing"
    model2:
      type: "gemini-pro"
      role: "assistant"
      instructions:
        template: "ai_assistant_instructions"
        params:
          domain: "Quantum Computing"
  goal: "Explore quantum computing concepts"
  timeouts:
    request: 300
    retry_count: 3
    notify_on:
      - timeout
      - retry
      - error
"""
    config_path.write_text(config_content)
    config = load_config(str(config_path))

    assert isinstance(config, DiscussionConfig)
    assert config.turns == 3
    assert len(config.models) == 2
    assert config.goal == "Explore quantum computing concepts"
    assert config.timeouts.request == 300


def test_invalid_model_type(tmp_path):
    """Test configuration with invalid model type"""
    config_path = tmp_path / "test_config.yaml"
    config_content = """
discussion:
  turns: 3
  models:
    model1:
      type: "invalid-model"
      role: "human"
  goal: "Test goal"
"""
    config_path.write_text(config_content)

    with pytest.raises(ValueError, match="Unsupported model type"):
        load_config(str(config_path))


def test_invalid_role(tmp_path):
    """Test configuration with invalid role"""
    config_path = tmp_path / "test_config.yaml"
    config_content = """
discussion:
  turns: 3
  models:
    model1:
      type: "claude-3-sonnet"
      role: "invalid"
  goal: "Test goal"
"""
    config_path.write_text(config_content)

    with pytest.raises(ValueError, match="Invalid role"):
        load_config(str(config_path))


def test_file_config(tmp_path):
    """Test file configuration validation"""
    # Create a test image file
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"dummy image content")

    config_path = tmp_path / "test_config.yaml"
    config_content = f"""
discussion:
  turns: 3
  models:
    model1:
      type: "claude-3-sonnet"
      role: "human"
    model2:
      type: "gemini-pro-vision"
      role: "assistant"
  goal: "Analyze the image"
  input_file:
    path: "{str(image_path)}"
    type: "image"
    max_resolution: "4096x4096"
"""
    config_path.write_text(config_content)
    config = load_config(str(config_path))

    assert isinstance(config.input_file, FileConfig)
    assert config.input_file.type == "image"
    assert config.input_file.path == str(image_path)


def test_timeout_config():
    """Test timeout configuration validation"""
    # Valid config
    timeout = TimeoutConfig(request=300, retry_count=3, notify_on=["timeout", "retry"])
    assert timeout.request == 300
    assert timeout.retry_count == 3

    # Invalid request timeout
    with pytest.raises(ValueError, match="Request timeout must be between"):
        TimeoutConfig(request=10)

    # Invalid retry count
    with pytest.raises(ValueError, match="Retry count must be between"):
        TimeoutConfig(retry_count=10)

    # Invalid notification events
    with pytest.raises(ValueError, match="Invalid notification events"):
        TimeoutConfig(notify_on=["invalid"])


def test_model_capabilities():
    """Test model capability detection"""
    # Vision model
    vision_model = ModelConfig(type="gemini-pro-vision", role="assistant")
    capabilities = detect_model_capabilities(vision_model)
    assert capabilities["vision"] is True

    # Standard model
    standard_model = ModelConfig(type="claude-3-sonnet", role="human")
    capabilities = detect_model_capabilities(standard_model)
    assert capabilities["vision"] is False
    assert capabilities["streaming"] is True
    assert capabilities["function_calling"] is True


def test_system_instructions(tmp_path):
    """Test loading and applying system instructions"""
    # Create a dummy system instructions file
    instructions_dir = tmp_path / "docs"
    instructions_dir.mkdir()
    instructions_path = instructions_dir / "system_instructions.md"
    instructions_content = """
# Test Instructions

```yaml
test_template:
  core: |
    Test instruction with {domain}
```
"""
    instructions_path.write_text(instructions_content)

    config_path = tmp_path / "test_config.yaml"
    config_content = """
discussion:
  turns: 3
  models:
    model1:
      type: "claude-3-sonnet"
      role: "human"
      instructions:
        template: "test_template"
        params:
          domain: "test domain"
  goal: "Test goal"
"""
    config_path.write_text(config_content)

    # Temporarily override the system instructions path
    import configuration

    original_path = configuration.Path
    configuration.Path = lambda x: tmp_path / x

    try:
        config = load_config(str(config_path))
        assert "test domain" in str(config.models["model1"].persona)
    finally:
        configuration.Path = original_path
