import streamlit as st
import subprocess
import sys
import os
import yaml
import tempfile
import re
import time
import base64
from PIL import Image
from io import BytesIO
import json
import uuid
from datetime import datetime

# Helper function to convert PIL image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Assuming examples/run_vision_discussion.py is in the examples directory
RUN_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "examples", "run_vision_discussion.py")

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="AI Battle Orchestrator")

# Load CSS from external file
with open("static/css/aibattle.css") as f:
    css_content = f.read()

# Add custom JavaScript for model selection
js_code = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Handle model tag clicks more reliably
    function setupModelTags() {
        // Setup for human model tags
        var humanTags = document.querySelectorAll('#human-model-container .model-tag');
        humanTags.forEach(function(tag) {
            tag.addEventListener('click', function() {
                var model = this.getAttribute('data-model');
                var input = document.getElementById('human_model_input');
                if (input) {
                    input.value = 'human_' + model;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                }
            });
        });

        // Setup for AI model tags
        var aiTags = document.querySelectorAll('#ai-model-container .model-tag');
        aiTags.forEach(function(tag) {
            tag.addEventListener('click', function() {
                var model = this.getAttribute('data-model');
                var input = document.getElementById('ai_model_input');
                if (input) {
                    input.value = 'ai_' + model;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                }
            });
        });
    }

    // Initial setup
    setupModelTags();

    // Setup mutation observer to handle dynamically added elements
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length) {
                setupModelTags();
            }
        });
    });

    // Start observing the document with the configured parameters
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
"""

# Combine CSS and JavaScript
st.markdown(f"<style>{css_content}</style>{js_code}", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'raw_log_lines' not in st.session_state:
    st.session_state.raw_log_lines = []
if 'config_mode' not in st.session_state:
    st.session_state.config_mode = "file"  # or "builder"
if 'show_logs' not in st.session_state:
    st.session_state.show_logs = True
if 'conversation_html' not in st.session_state:
    st.session_state.conversation_html = ""
if 'battle_id' not in st.session_state:
    st.session_state.battle_id = str(uuid.uuid4())
if 'has_disagreement' not in st.session_state:
    st.session_state.has_disagreement = False
if 'model_status' not in st.session_state:
    st.session_state.model_status = {}
if 'last_activity' not in st.session_state:
    st.session_state.last_activity = time.time()

# Helper functions
def toggle_logs():
    st.session_state.show_logs = not st.session_state.show_logs

def switch_config_mode(mode):
    st.session_state.config_mode = mode

def display_image(img_path_or_base64):
    try:
        # Check if it's a base64 string
        if isinstance(img_path_or_base64, str) and img_path_or_base64.startswith('data:image'):
            # Extract the base64 part
            base64_data = img_path_or_base64.split(',')[1]
            img = Image.open(BytesIO(base64.b64decode(base64_data)))
            return img
        # Check if it's a file path
        elif isinstance(img_path_or_base64, str) and os.path.exists(img_path_or_base64):
            return Image.open(img_path_or_base64)
        # BytesIO object
        elif isinstance(img_path_or_base64, BytesIO):
            return Image.open(img_path_or_base64)
        else:
            return None
    except Exception as e:
        st.error(f"Error displaying image: {e}")
        return None

def render_message_with_thinking(content):
    # Find all thinking tags in the content
    thinking_pattern = r'<thinking>(.*?)</thinking>'
    thinking_sections = re.findall(thinking_pattern, content, re.DOTALL)

    # Remove thinking sections from the main content
    clean_content = re.sub(thinking_pattern, '', content, flags=re.DOTALL)

    # Return both the clean content and thinking sections
    return clean_content.strip(), thinking_sections

def render_with_images(content, uploaded_images):
    # Simple image markdown detection - can be expanded for other formats
    image_pattern = r'!\[.*?\]\((.*?)\)'

    # Find all image references
    image_matches = re.findall(image_pattern, content)

    # Replace image markdown with placeholders
    content_with_placeholders = re.sub(image_pattern, '{{IMAGE_PLACEHOLDER}}', content)

    # Split content by image placeholders
    content_parts = content_with_placeholders.split('{{IMAGE_PLACEHOLDER}}')

    # Prepare image data
    image_data = []
    for img_ref in image_matches:
        # Check if it's a reference to an uploaded image
        if img_ref in uploaded_images:
            image_data.append(uploaded_images[img_ref])
        else:
            # Try to load from path
            try:
                if os.path.exists(img_ref):
                    with open(img_ref, "rb") as f:
                        image_data.append(BytesIO(f.read()))
                else:
                    image_data.append(None)
            except Exception as e:
                st.error(f"Error loading image {img_ref}: {e}")
                image_data.append(None)

    return content_parts, image_data

def export_conversation(conversation_history, raw_logs, battle_id):
    """Export conversation history and logs to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_data = {
        "battle_id": battle_id,
        "timestamp": timestamp,
        "conversation": conversation_history,
        "raw_logs": raw_logs
    }

    # Generate a filename
    filename = f"aibattle_export_{timestamp}_{battle_id[:8]}.json"

    # Convert to JSON string
    json_str = json.dumps(export_data, indent=2)

    # Create a download button
    st.download_button(
        label=f"Export Conversation",
        data=json_str,
        file_name=filename,
        mime="application/json",
        key=f"export_{timestamp}"
    )

# Function to get available models from ai-battle.py
def get_available_models():
    """Extract available models from ai-battle.py"""
    try:
        # First approach: Try to import directly
        try:
            import sys
            import os
            import importlib.util

            # Add parent directory to path if needed
            sys.path.append(os.path.dirname(__file__))

            # First try direct import
            try:
                from ai_battle import OPENAI_MODELS, CLAUDE_MODELS, GEMINI_MODELS, ANTHROPIC_MODELS, ConversationManager

                # Create a temporary manager to get models
                temp_manager = ConversationManager(domain="Model Detection")
                return temp_manager.get_available_models()
            except ImportError:
                # Try to parse the file directly if module import fails
                ai_battle_path = os.path.join(os.path.dirname(__file__), "ai-battle.py")
                if os.path.exists(ai_battle_path):
                    # Extract model lists from file content
                    with open(ai_battle_path, 'r') as f:
                        content = f.read()

                    # Use regex to find model definitions
                    import re

                    # Dictionary to store models
                    models = {
                        "openai": [],
                        "claude": [],
                        "anthropic": [],
                        "gemini": [],
                        "ollama": [],
                        "lmstudio": [],
                        "local": []
                    }

                    # Extract OpenAI models
                    openai_match = re.search(r'OPENAI_MODELS\s*=\s*\[(.*?)\]', content, re.DOTALL)
                    if openai_match:
                        openai_str = openai_match.group(1)
                        openai_models = re.findall(r'"([^"]+)"', openai_str)
                        models["openai"] = openai_models

                    # Extract Claude models
                    claude_match = re.search(r'CLAUDE_MODELS\s*=\s*\[(.*?)\]', content, re.DOTALL)
                    if claude_match:
                        claude_str = claude_match.group(1)
                        claude_models = re.findall(r'"([^"]+)"', claude_str)
                        models["claude"] = claude_models

                    # Extract Anthropic models
                    anthropic_match = re.search(r'ANTHROPIC_MODELS\s*=\s*\[(.*?)\]', content, re.DOTALL)
                    if anthropic_match:
                        anthropic_str = anthropic_match.group(1)
                        anthropic_models = re.findall(r'"([^"]+)"', anthropic_str)
                        models["anthropic"] = anthropic_models

                    # Extract Gemini models
                    gemini_match = re.search(r'GEMINI_MODELS\s*=\s*\[(.*?)\]', content, re.DOTALL)
                    if gemini_match:
                        gemini_str = gemini_match.group(1)
                        gemini_models = re.findall(r'"([^"]+)"', gemini_str)
                        models["gemini"] = gemini_models

                    # Extract Ollama models
                    ollama_match = re.search(r'OLLAMA_MODELS\s*=\s*\[(.*?)\]', content, re.DOTALL)
                    if ollama_match:
                        ollama_str = ollama_match.group(1)
                        ollama_models = re.findall(r'"([^"]+)"', ollama_str)
                        models["ollama"] = ollama_models

                    return models
                else:
                    raise FileNotFoundError(f"Could not find ai-battle.py at {ai_battle_path}")
        except Exception as inner_e:
            st.warning(f"Inner error loading models: {inner_e}")
            # Fallback to hardcoded models list
            models = {
                "openai": ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4-1106-preview", "gpt-4-vision-preview"],
                "claude": ["claude-3-7-sonnet", "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"],
                "anthropic": ["claude-3-sonnet", "claude-3-haiku", "claude-3-opus"],
                "gemini": ["gemini-2.5-flash-preview", "gemini-2.5-pro-exp", "gemini-2.0-pro", "gemini-1.5-pro"],
                "ollama": ["ollama-llama3", "ollama-mixtral", "ollama-mistral"],
                "lmstudio": ["lmstudio-model"],
                "local": ["mlx-llama-3.1-abb"]
            }
            return models
    except Exception as e:
        st.warning(f"Error loading model list: {e}")
        # Fallback to basic model list
        return {
            "openai": ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4-1106-preview", "gpt-4-vision-preview"],
            "claude": ["claude-3-7-sonnet", "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"],
            "anthropic": ["claude-3-sonnet", "claude-3-haiku", "claude-3-opus"],
            "gemini": ["gemini-2.5-flash-preview", "gemini-2.5-pro-exp", "gemini-2.0-pro", "gemini-1.5-pro"],
            "ollama": ["ollama-llama3", "ollama-mixtral", "ollama-mistral"],
            "lmstudio": ["lmstudio-model"],
            "local": ["mlx-llama-3.1-abb"]
        }

def detect_disagreement(conversation_history):
    """Simple heuristic to detect disagreement between models"""
    if len(conversation_history) < 2:
        return False

    # Look for disagreement keywords in the conversation
    disagreement_keywords = [
        "disagree", "incorrect", "not accurate", "wrong", "mistaken",
        "error", "misunderstood", "actually", "contrary", "oppose",
        "differs", "misleading", "not true"
    ]

    for i in range(1, len(conversation_history)):
        current_msg = conversation_history[i]["content"].lower()

        # Check if any disagreement keywords appear in the message
        if any(keyword in current_msg for keyword in disagreement_keywords):
            return True

    return False

# Title
st.title("AI Battle Orchestrator")
st.write("Configure and run AI model conversations with advanced UI features.")

# Configuration container
with st.container():
    st.markdown('<div class="config-container">', unsafe_allow_html=True)

    # Custom tabs for config mode
    # Create tabs for configuration method
    config_tab_file, config_tab_builder = st.tabs(["Config File", "Config Builder"])

    # Set the active tab based on the session state
    if st.session_state.config_mode == "file":
        with config_tab_file:
            st.session_state.config_mode = "file"
    else:
        with config_tab_builder:
            st.session_state.config_mode = "builder"

    # Config File Option - Inside the file tab
    with config_tab_file:
        st.session_state.config_mode = "file"

        # Required Configuration File
        config_file = st.file_uploader("Upload Configuration File (Required):", type=["yaml", "json"])

        # Initial Prompt Override (for all tabs)
        st.subheader("Goal/Initial Prompt Override")
        initial_prompt_override = st.text_area(
            "Enter Goal/Initial Prompt (will override config file):",
            height=100,
            help="This will override any goal/prompt in the configuration file"
        )

        # Display configuration preview if a file is uploaded
        if config_file is not None:
            try:
                # Store the parsed config in session state so it persists
                config_content = config_file.getvalue().decode('utf-8')
                if config_file.name.endswith('.json'):
                    config_data = json.loads(config_content)
                else:
                    config_data = yaml.safe_load(config_content)

                # Store in session state
                st.session_state.current_config = config_data

                # Show the configuration in an expander
                with st.expander("Configuration Preview", expanded=True):
                    st.code(yaml.dump(config_data, default_flow_style=False), language='yaml')

                # Add button to copy to builder
                if st.button("Copy to Config Builder"):
                    # Switch to builder tab
                    st.session_state.config_mode = "builder"

                    # Copy relevant fields if available
                    if "discussion" in config_data:
                        discussion = config_data["discussion"]

                        # Copy goal/prompt
                        if "goal" in discussion:
                            st.session_state.initial_prompt = discussion["goal"]

                        # Copy models
                        if "models" in discussion:
                            for model_key, model_data in discussion["models"].items():
                                if "type" in model_data and "role" in model_data:
                                    role = model_data["role"]
                                    model_type = model_data["type"]

                                    if role == "human":
                                        st.session_state.human_model = model_type
                                        if "persona" in model_data:
                                            st.session_state.human_persona = model_data["persona"]
                                    elif role == "assistant":
                                        st.session_state.ai_model = model_type
                                        if "persona" in model_data:
                                            st.session_state.ai_persona = model_data["persona"]

                        # Copy rounds
                        if "turns" in discussion:
                            st.session_state.num_rounds = discussion["turns"]

                # Model Override Section
                st.subheader("Override Models in Configuration")
                col1, col2 = st.columns(2)

                # Extract original models from config
                orig_human_model = "default"
                orig_ai_model = "default"
                try:
                    if "discussion" in config_data and "models" in config_data["discussion"]:
                        models = config_data["discussion"]["models"]
                        if "model1" in models and "type" in models["model1"]:
                            orig_human_model = models["model1"]["type"]
                        if "model2" in models and "type" in models["model2"]:
                            orig_ai_model = models["model2"]["type"]

                        with col1:
                            st.write(f"Original Human Model: **{orig_human_model}**")
                            override_human = st.checkbox("Override Human Model", value=False)
                            if override_human:
                                use_custom_human = st.checkbox("Use custom human model name", value=False, key="custom_human_file")

                                if use_custom_human:
                                    # Free text input for custom model
                                    custom_human_model = st.text_input("Custom Human Model Name:",
                                                               placeholder="e.g., ollama-phi3:3b or any model name",
                                                               key="custom_human_input_file")
                                    # Save to session state
                                    st.session_state.human_model = custom_human_model
                                else:
                                    # Get model categories
                                    model_categories = get_available_models()
                                    # Flatten into a single list for simplicity
                                    all_models = []
                                    for category, models in model_categories.items():
                                        all_models.extend(models)

                                    new_human_model = st.selectbox("New Human Model", all_models, index=0)
                                    # Save to session state
                                    st.session_state.human_model = new_human_model
                            else:
                                st.session_state.human_model = orig_human_model

                        with col2:
                            st.write(f"Original Assistant Model: **{orig_ai_model}**")
                            override_ai = st.checkbox("Override Assistant Model", value=False)
                            if override_ai:
                                use_custom_ai = st.checkbox("Use custom assistant model name", value=False, key="custom_ai_file")

                                if use_custom_ai:
                                    # Free text input for custom model
                                    custom_ai_model = st.text_input("Custom Assistant Model Name:",
                                                            placeholder="e.g., ollama-llama3:8b or any model name",
                                                            key="custom_ai_input_file")
                                    # Save to session state
                                    st.session_state.ai_model = custom_ai_model
                                else:
                                    # Get model categories
                                    model_categories = get_available_models()
                                    # Flatten into a single list for simplicity
                                    all_models = []
                                    for category, models in model_categories.items():
                                        all_models.extend(models)

                                    new_ai_model = st.selectbox("New Assistant Model", all_models, index=0)
                                    # Save to session state
                                    st.session_state.ai_model = new_ai_model
                            else:
                                st.session_state.ai_model = orig_ai_model
                except Exception as e:
                    st.warning(f"Could not extract model information from config: {e}")
            except Exception as e:
                st.error(f"Error parsing configuration file: {e}")

        # Custom model section
        st.subheader("Custom Model Names")
        custom_human_model_enabled = st.checkbox("Use custom human model name", value=False)
        if custom_human_model_enabled:
            custom_human_model = st.text_input("Custom Human Model Name:",
                                          placeholder="e.g., ollama-phi3:3b or any model name")
            st.session_state.human_model = custom_human_model

        custom_ai_model_enabled = st.checkbox("Use custom assistant model name", value=False)
        if custom_ai_model_enabled:
            custom_ai_model = st.text_input("Custom Assistant Model Name:",
                                       placeholder="e.g., ollama-llama3:8b or any model name")
            st.session_state.ai_model = custom_ai_model

    # Config Builder Option - Inside the builder tab
    with config_tab_builder:
        st.session_state.config_mode = "builder"
        # Get available models from ai-battle.py
        model_categories = get_available_models()
        all_models = []
        for category, models in model_categories.items():
            all_models.extend(models)

        # Initialize model selection in session state if not present
        if 'human_model' not in st.session_state:
            st.session_state.human_model = next(iter(model_categories["gemini"]), "gemini-2.5-flash-preview") if model_categories["gemini"] else "gemini-2.5-flash-preview"
        if 'ai_model' not in st.session_state:
            st.session_state.ai_model = next(iter(model_categories["claude"]), "claude-3-7-sonnet") if model_categories["claude"] else "claude-3-7-sonnet"

        # Function to handle model selection with proper prefix handling
        def select_human_model(model_name):
            if model_name is None:
                return

            if isinstance(model_name, str) and model_name.startswith('human_'):
                model_name = model_name[6:]  # Remove "human_" prefix

            if not st.session_state.get('use_custom_human_model', False):
                st.session_state.human_model = model_name

        def select_ai_model(model_name):
            if model_name is None:
                return

            if isinstance(model_name, str) and model_name.startswith('ai_'):
                model_name = model_name[3:]  # Remove "ai_" prefix

            if not st.session_state.get('use_custom_ai_model', False):
                st.session_state.ai_model = model_name

        st.subheader("Model Selection")

        # Custom model options
        custom_cols = st.columns(2)
        with custom_cols[0]:
            use_custom_human = st.checkbox("Use custom human model", key="use_custom_human_model", value=False)
            if use_custom_human:
                custom_human_model = st.text_input("Custom Human Model Name:",
                                            placeholder="e.g., ollama-phi3:3b or any model name",
                                            key="custom_human_builder")
                if custom_human_model:
                    st.session_state.human_model = custom_human_model

        with custom_cols[1]:
            use_custom_ai = st.checkbox("Use custom assistant model", key="use_custom_ai_model", value=False)
            if use_custom_ai:
                custom_ai_model = st.text_input("Custom Assistant Model Name:",
                                         placeholder="e.g., ollama-llama3:8b or any model name",
                                         key="custom_ai_builder")
                if custom_ai_model:
                    st.session_state.ai_model = custom_ai_model

        # Human Model Selection - By Category
        st.write("Human Agent Model:")

        human_model_html = ""
        for category, models in model_categories.items():
            if not models:  # Skip empty categories
                continue

            human_model_html += f'<div class="model-category"><div class="model-category-title">{category.title()}</div>'
            human_model_html += '<div class="model-tag-container" id="human-model-container">'

            for model in models:
                selected_class = "selected" if model == st.session_state.human_model else ""
                # Add a data attribute to make it easier to identify in JavaScript
                human_model_html += f'<span class="model-tag {selected_class}" data-model="{model}" '
                human_model_html += f'onclick="document.getElementById(\'human_model_input\').value = \'human_{model}\'; document.getElementById(\'human_model_input\').dispatchEvent(new Event(\'input\', {{ bubbles: true }}))">{model}</span>'

            human_model_html += '</div></div>'

        st.markdown(human_model_html, unsafe_allow_html=True)

        # Add a fallback select box and hidden input for JavaScript
        st.markdown('<div id="model-selection-debug" style="display:none;">', unsafe_allow_html=True)
        human_model_selection = st.text_input("Human Model Selection:",
                                         key="human_model_input",
                                         label_visibility="visible")
        st.markdown('</div>', unsafe_allow_html=True)

        # Add a normal dropdown as fallback
        st.markdown("<p><small>Can't click the model tags? Use this dropdown instead:</small></p>", unsafe_allow_html=True)
        # Create a flat list of all available models
        all_human_models = [model for category, models in model_categories.items() for model in models]

        if all_human_models:  # Only create selectbox if we have models
            try:
                # Find the index of the current model or default to 0
                current_index = all_human_models.index(st.session_state.human_model) if st.session_state.human_model in all_human_models else 0

                fallback_human_selection = st.selectbox(
                    "Select Human Model",
                    options=all_human_models,
                    index=current_index,
                    key="fallback_human_model"
                )

                # Update the model explicitly
                if fallback_human_selection:
                    st.session_state.human_model = fallback_human_selection
            except Exception as e:
                st.warning(f"Error setting up human model selection: {e}")
                fallback_human_selection = None
        else:
            st.warning("No models available to select from")
            fallback_human_selection = None

        # Update from either source
        if human_model_selection and human_model_selection.startswith("human_"):
            select_human_model(human_model_selection[6:])
        else:
            # Update from fallback dropdown
            select_human_model(fallback_human_selection)

        # AI Model Selection - By Category
        st.write("Assistant Agent Model:")

        ai_model_html = ""
        for category, models in model_categories.items():
            if not models:  # Skip empty categories
                continue

            ai_model_html += f'<div class="model-category"><div class="model-category-title">{category.title()}</div>'
            ai_model_html += '<div class="model-tag-container" id="ai-model-container">'

            for model in models:
                selected_class = "selected" if model == st.session_state.ai_model else ""
                # Add a data attribute to make it easier to identify in JavaScript
                ai_model_html += f'<span class="model-tag {selected_class}" data-model="{model}" '
                ai_model_html += f'onclick="document.getElementById(\'ai_model_input\').value = \'ai_{model}\'; document.getElementById(\'ai_model_input\').dispatchEvent(new Event(\'input\', {{ bubbles: true }}))">{model}</span>'

            ai_model_html += '</div></div>'

        st.markdown(ai_model_html, unsafe_allow_html=True)

        # Add a fallback select box and hidden input for JavaScript
        st.markdown('<div id="ai-model-selection-debug" style="display:none;">', unsafe_allow_html=True)
        ai_model_selection = st.text_input("Assistant Model Selection:",
                                      key="ai_model_input",
                                      label_visibility="visible")
        st.markdown('</div>', unsafe_allow_html=True)

        # Add a normal dropdown as fallback
        st.markdown("<p><small>Can't click the model tags? Use this dropdown instead:</small></p>", unsafe_allow_html=True)
        # Create a flat list of all available models
        all_ai_models = [model for category, models in model_categories.items() for model in models]

        if all_ai_models:  # Only create selectbox if we have models
            try:
                # Find the index of the current model or default to 0
                current_index = all_ai_models.index(st.session_state.ai_model) if st.session_state.ai_model in all_ai_models else 0

                fallback_ai_selection = st.selectbox(
                    "Select Assistant Model",
                    options=all_ai_models,
                    index=current_index,
                    key="fallback_ai_model"
                )

                # Update the model explicitly
                if fallback_ai_selection:
                    st.session_state.ai_model = fallback_ai_selection
            except Exception as e:
                st.warning(f"Error setting up assistant model selection: {e}")
                fallback_ai_selection = None
        else:
            st.warning("No models available to select from")
            fallback_ai_selection = None

        # Update from either source
        if ai_model_selection and ai_model_selection.startswith("ai_"):
            select_ai_model(ai_model_selection[3:])
        else:
            # Update from fallback dropdown
            select_ai_model(fallback_ai_selection)

        col1, col2 = st.columns(2)

        with col1:
            human_role = st.text_input("Human Agent Role:", value="human")
            # Use value from session state if available
            human_persona_value = st.session_state.get('human_persona', "You are a visual analysis expert analyzing the content.")
            human_persona = st.text_area("Human Agent Persona:", height=100,
                                         value=human_persona_value)
            # Store in session state
            st.session_state.human_persona = human_persona

        with col2:
            ai_role = st.text_input("Assistant Agent Role:", value="assistant")
            # Use value from session state if available
            ai_persona_value = st.session_state.get('ai_persona', "You are an AI assistant analyzing visual content.")
            ai_persona = st.text_area("Assistant Agent Persona:", height=100,
                                      value=ai_persona_value)
            # Store in session state
            st.session_state.ai_persona = ai_persona

        # Store the selected models
        human_model = st.session_state.human_model
        ai_model = st.session_state.ai_model

        # Initial Prompt & Rounds
        # Initialize with session state value if one exists
        initial_prompt_value = st.session_state.get('initial_prompt', '')

        initial_prompt = st.text_area("Enter Goal/Initial Prompt:",
                                value=initial_prompt_value,
                                height=100,
                                placeholder="Enter the goal or instructions for the conversation here...")

        # Store in session state for persistence
        st.session_state.initial_prompt = initial_prompt
        num_rounds = st.number_input("Number of Rounds:", min_value=1, value=2)

        # Timeouts
        with st.expander("Advanced Options"):
            request_timeout = st.number_input("Request Timeout (seconds):", min_value=30, value=400)
            retry_count = st.number_input("Retry Count:", min_value=0, value=2)
            notify_options = st.multiselect("Notify On:",
                                            options=["timeout", "retry", "error"],
                                            default=["timeout", "retry", "error"])

    # Image Upload (for both modes)
    st.subheader("Input Files")
    uploaded_images = st.file_uploader("Upload Image(s):", type=["png", "jpg", "jpeg", "gif"], accept_multiple_files=True)

    # Convert uploaded images to a dict with paths as keys
    image_files = {}
    image_paths = []

    if uploaded_images:
        for i, img_file in enumerate(uploaded_images):
            # Create a temporary file for each uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{img_file.name}") as tmp_file:
                tmp_file.write(img_file.getvalue())
                img_path = tmp_file.name
                image_paths.append(img_path)
                # Store the image data
                image_files[img_path] = BytesIO(img_file.getvalue())

                # Show preview
                st.image(img_file, caption=f"Image {i+1}: {img_file.name}", width=200)

    st.markdown('</div>', unsafe_allow_html=True)

# Create containers for the main content
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Conversation container
conversation_container = st.container()
with conversation_container:
    st.subheader("Conversation")
    st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
    conversation_area = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# Flow chart container
with st.container():
    st.subheader("Conversation Flow")
    st.markdown('<div class="flow-chart-container">', unsafe_allow_html=True)
    flow_chart_area = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Create fixed footer container for logs
st.markdown('<div class="footer-container">', unsafe_allow_html=True)

# Create a toggle button for logs
st.markdown(f"""
<div class="log-container-header">
    <span>Terminal Output (Logs)</span>
    <button class="log-toggle-btn" onclick="document.getElementById('toggle_logs_btn').click()">
        {"▼ Hide" if st.session_state.show_logs else "▲ Show"}
    </button>
</div>
""", unsafe_allow_html=True)

# Create a button for the toggle interaction (hidden, just for JS to click)
toggle_logs_btn = st.button("Toggle Logs", key="toggle_logs_btn", type="secondary")
if toggle_logs_btn:
    toggle_logs()

# Create a designated placeholder for logs
log_placeholder = st.empty()

# Always create the container, but populate it only if shown
# This ensures the DOM element always exists and just toggles visibility
log_container_html = '<div class="log-container" style="{}">'.format("display:block" if st.session_state.show_logs else "display:none")
log_container_html += f'<pre>{"".join(st.session_state.raw_log_lines)}</pre></div>'
log_placeholder.markdown(log_container_html, unsafe_allow_html=True)

# Close the fixed footer container
st.markdown('</div>', unsafe_allow_html=True)

# Start Button
start_col1, start_col2 = st.columns([4, 1])
with start_col1:
    start_button = st.button("Start AI Battle", use_container_width=True)

with start_col2:
    if len(st.session_state.conversation_history) > 0:
        export_button = st.button("Export Results", use_container_width=True)
        if export_button:
            export_conversation(
                st.session_state.conversation_history,
                st.session_state.raw_log_lines,
                st.session_state.battle_id
            )

if start_button:
    config_valid = False

    # Check for initial prompt override first as it applies to both modes
    has_prompt_override = bool(initial_prompt_override and initial_prompt_override.strip())

    if st.session_state.config_mode == "file":
        if config_file is None:
            st.warning("Please upload a configuration file.")
        else:
            # If we have a prompt override, we're good to go
            if has_prompt_override:
                config_valid = True
            else:
                # If no override, check that the config has a goal
                try:
                    config_content = config_file.getvalue().decode('utf-8')
                    if config_file.name.endswith('.json'):
                        config_data = json.loads(config_content)
                    else:
                        config_data = yaml.safe_load(config_content)

                    # Validate the config has a discussion.goal
                    if "discussion" in config_data and "goal" in config_data["discussion"] and config_data["discussion"]["goal"]:
                        config_valid = True
                    else:
                        st.warning("Configuration file missing valid 'discussion.goal'. Please add a goal or use the override field.")
                except Exception as e:
                    st.error(f"Error validating configuration file: {e}")
    else:  # Builder mode
        # Accept either the builder prompt or the override
        if not initial_prompt and not has_prompt_override:
            st.warning("Please enter a goal/initial prompt in the builder or in the override field.")
        else:
            config_valid = True

    if config_valid:
        st.info("Starting AI Battle...")

        # Reset state for a new run
        st.session_state.conversation_history = []
        st.session_state.raw_log_lines = []
        st.session_state.conversation_html = ""
        st.session_state.battle_id = str(uuid.uuid4())
        st.session_state.has_disagreement = False

        # Create a temporary config file
        temp_config_path = None
        try:
            if st.session_state.config_mode == "file":
                # Use uploaded config file, but with model overrides if specified
                config_content = config_file.getvalue().decode('utf-8')
                if config_file.name.endswith('.json'):
                    config_data = json.loads(config_content)
                else:
                    config_data = yaml.safe_load(config_content)

                # Apply model overrides if models were changed in the UI
                if hasattr(st.session_state, "human_model") and hasattr(st.session_state, "ai_model"):
                    model_was_overridden = False

                    # Check if this is an image/file config - if so, force ai-ai mode
                    has_images = False
                    if "discussion" in config_data:
                        if "input_file" in config_data["discussion"]:
                            file_config = config_data["discussion"]["input_file"]
                            if isinstance(file_config, dict) and file_config.get("type") == "image":
                                has_images = True
                        if "input_files" in config_data["discussion"]:
                            files_config = config_data["discussion"]["input_files"]
                            if isinstance(files_config, dict) and "files" in files_config:
                                has_images = True

                    # Force ai-ai mode for images
                    if has_images:
                        st.info("Images detected in configuration - forcing ai-ai mode for proper vision support")
                        # Set explicit ai-ai mode in configuration
                        config_data["discussion"]["mode"] = "ai-ai"

                    if "discussion" in config_data and "models" in config_data["discussion"]:
                        if st.session_state.human_model != "default":
                            # Override human model
                            for model_key, model_data in config_data["discussion"]["models"].items():
                                if model_data.get("role") == "human":
                                    config_data["discussion"]["models"][model_key]["type"] = st.session_state.human_model
                                    st.info(f"Overriding human model to: {st.session_state.human_model}")
                                    model_was_overridden = True

                        if st.session_state.ai_model != "default":
                            # Override AI model
                            for model_key, model_data in config_data["discussion"]["models"].items():
                                if model_data.get("role") == "assistant":
                                    config_data["discussion"]["models"][model_key]["type"] = st.session_state.ai_model
                                    st.info(f"Overriding assistant model to: {st.session_state.ai_model}")
                                    model_was_overridden = True

                # Write the modified config to a temp file
                temp_config_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
                temp_config_path = temp_config_file.name
                with open(temp_config_path, "w") as f:
                    yaml.dump(config_data, f)

                # Check if we have an initial prompt override
                prompt_was_overridden = False
                if has_prompt_override and "discussion" in config_data:
                    config_data["discussion"]["goal"] = initial_prompt_override
                    st.info(f"Overriding goal/prompt in configuration")
                    prompt_was_overridden = True

                if model_was_overridden and prompt_was_overridden:
                    st.success(f"Using uploaded config file: {config_file.name} with model and prompt overrides")
                elif model_was_overridden:
                    st.success(f"Using uploaded config file: {config_file.name} with model overrides")
                elif prompt_was_overridden:
                    st.success(f"Using uploaded config file: {config_file.name} with prompt override")
                else:
                    st.success(f"Using uploaded config file: {config_file.name}")
            else:
                # Create a config based on builder inputs - use override if available
                goal_text = initial_prompt_override.strip() if has_prompt_override else initial_prompt
                default_config = {
                    "discussion": {
                        "goal": goal_text,
                        "turns": num_rounds,
                        "models": {
                            "model1": {
                                "type": human_model,
                                "role": human_role,
                                "persona": human_persona
                            },
                            "model2": {
                                "type": ai_model,
                                "role": ai_role,
                                "persona": ai_persona
                            }
                        },
                        "timeouts": {
                            "request": request_timeout,
                            "retry_count": retry_count,
                            "notify_on": notify_options
                        }
                    }
                }

                # Add image configuration if images were uploaded
                if image_paths:
                    # If we have images, ALWAYS use ai-ai mode to ensure both participants get the images
                    mode = "ai-ai"  # Force ai-ai mode for image discussions

                    if len(image_paths) == 1:
                        # Single image - use input_file
                        default_config["discussion"]["input_file"] = {
                            "path": image_paths[0],
                            "type": "image",
                            "max_resolution": "1024x1024"
                        }
                    else:
                        # Multiple images - use input_files with proper structure
                        # Create the file config objects first
                        file_configs = [
                            {"path": path, "type": "image", "max_resolution": "1024x1024"}
                            for path in image_paths
                        ]

                        # Then use the proper MultiFileConfig structure expected by run_vision_discussion.py
                        default_config["discussion"]["input_files"] = {
                            "files": file_configs
                        }

                temp_config_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
                temp_config_path = temp_config_file.name
                with open(temp_config_path, "w") as f:
                    yaml.dump(default_config, f)
                st.info("Using generated config from builder interface.")

            # Define the command to run examples/run_vision_discussion.py
            command = [
                sys.executable,  # Use the current Python executable
                RUN_SCRIPT_PATH,
                temp_config_path,
            ]

            # Initialize conversation HTML and model status
            conversation_html = ""
            conversation_area.markdown(conversation_html, unsafe_allow_html=True)
            st.session_state.model_status = {}

            # Create a status area for model activity
            status_placeholder = st.empty()

            # Status update function
            def update_model_status():
                status_html = '<div class="model-status-container">'
                status_html += '<h4>Model Status:</h4>'

                if not st.session_state.model_status:
                    status_html += '<p>Waiting for models to initialize...</p>'

                current_time = time.time()
                for model, status in st.session_state.model_status.items():
                    # Calculate time elapsed
                    elapsed = current_time - status.get("timestamp", current_time)
                    status_class = "status-active" if status.get("active", False) else "status-waiting"

                    status_html += f'<div class="model-status {status_class}">'
                    status_html += f'<span class="model-name">{model}</span>: '

                    if status.get("error"):
                        status_html += f'<span class="status-error">Error: {status["error"]}</span>'
                    elif status.get("active", False):
                        status_html += f'<span class="status-active">Processing ({elapsed:.1f}s)</span>'
                    else:
                        status_html += f'<span class="status-done">Done</span>'

                    status_html += '</div>'

                status_html += '</div>'
                status_placeholder.markdown(status_html, unsafe_allow_html=True)

            # Initial status update
            update_model_status()

            # Create shared variables for the keepalive mechanism
            import threading

            # Create a thread-safe object to store last activity time
            # This avoids accessing session_state from a background thread
            class ThreadSafeActivityMonitor:
                def __init__(self):
                    self.last_activity = time.time()
                    self.lock = threading.Lock()

                def update(self):
                    with self.lock:
                        self.last_activity = time.time()

                def get_time_since_last_activity(self):
                    with self.lock:
                        return time.time() - self.last_activity

            # Initialize the activity monitor
            activity_monitor = ThreadSafeActivityMonitor()

            # Update with current activity
            activity_monitor.update()

            # Create a keepalive mechanism
            def start_keepalive():
                while True:
                    # Check for activity timeout (5 min)
                    time_since_activity = activity_monitor.get_time_since_last_activity()

                    if time_since_activity > 300:
                        # If no activity for 5 minutes, create a flag file to signal timeout
                        # We'll check for this file in the main thread
                        try:
                            with open("model_timeout_flag.txt", "w") as f:
                                f.write("timeout")
                        except:
                            pass  # Ignore any errors writing the flag file

                    time.sleep(1)  # Update every second

            # Start keepalive in a separate thread
            import threading
            keepalive_thread = threading.Thread(target=start_keepalive, daemon=True)
            keepalive_thread.start()

            # Run the script and capture output in real-time
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            # Display output in real-time and parse conversation turns
            current_turn_content = ""
            current_turn_role = None

            # Process output line by line
            for line in process.stdout:
                # Check if this line is a user/assistant message that should go to chat
                is_conversation_line = False

                # Add to raw log lines
                st.session_state.raw_log_lines.append(line)

                # Update last activity timestamp in both places
                st.session_state.last_activity = time.time()
                activity_monitor.update()

                # Check for timeout flag from the background thread
                if os.path.exists("model_timeout_flag.txt"):
                    try:
                        # Remove the flag file
                        os.remove("model_timeout_flag.txt")

                        # Mark models as error
                        for model in st.session_state.model_status:
                            if st.session_state.model_status[model].get("active", False):
                                st.session_state.model_status[model]["active"] = False
                                st.session_state.model_status[model]["error"] = "Timeout after 5 minutes of inactivity"

                        # Update the status display
                        update_model_status()
                    except:
                        pass  # Ignore errors removing the file

                # Check for model processing status
                model_start_match = re.search(r"Processing with model: ([^\s]+)", line)
                if model_start_match:
                    model_name = model_start_match.group(1)
                    st.session_state.model_status[model_name] = {
                        "active": True,
                        "timestamp": time.time()
                    }
                    update_model_status()

                # Check for model completion
                model_done_match = re.search(r"Completed processing with model: ([^\s]+)", line)
                if model_done_match:
                    model_name = model_done_match.group(1)
                    if model_name in st.session_state.model_status:
                        st.session_state.model_status[model_name]["active"] = False
                        update_model_status()

                # Check for model errors
                model_error_match = re.search(r"Error with model ([^\s]+): (.*)", line)
                if model_error_match:
                    model_name = model_error_match.group(1)
                    error_msg = model_error_match.group(2)
                    st.session_state.model_status[model_name] = {
                        "active": False,
                        "error": error_msg,
                        "timestamp": time.time()
                    }
                    update_model_status()

                # Check for the start of a new conversation turn
                turn_start_match = re.match(r"(USER|ASSISTANT|SYSTEM|HUMAN|AI): (.*)", line)

                if turn_start_match:
                    is_conversation_line = True
                    # If we were collecting content for a previous turn, save and display it
                    if current_turn_role is not None and current_turn_content:
                        # Map role names to standardized roles
                        role_mapping = {
                            "USER": "human",
                            "HUMAN": "human",
                            "ASSISTANT": "assistant",
                            "AI": "assistant",
                            "SYSTEM": "system"
                        }

                        std_role = role_mapping.get(current_turn_role.upper(), "system")
                        st.session_state.conversation_history.append({
                            "role": std_role,
                            "content": current_turn_content.strip()
                        })

                        # Process the content to extract thinking sections
                        clean_content, thinking_sections = render_message_with_thinking(current_turn_content.strip())

                        # Process content to find images
                        content_parts, image_data = render_with_images(clean_content, image_files)

                        # Create HTML for this message
                        avatar_emoji = "🧑‍💻" if std_role == "human" else ("🤖" if std_role == "assistant" else "ℹ️")
                        avatar_class = "human-avatar" if std_role == "human" else "assistant-avatar"
                        bubble_class = "human-bubble" if std_role == "human" else "assistant-bubble"
                        row_class = f"message-row {std_role}"

                        message_html = ""

                        # Add thinking sections BEFORE message bubble
                        for thinking in thinking_sections:
                            # Clean up the thinking content
                            thinking = thinking.strip()
                            message_html += f'<div class="thinking-section"><strong>Thinking:</strong><pre>{thinking}</pre></div>'

                        # Add the message bubble
                        message_html += f'<div class="{row_class}">'
                        message_html += f'<div class="avatar {avatar_class}">{avatar_emoji}</div>'
                        message_html += f'<div class="{bubble_class}">'

                        # Add content with images
                        for i, part in enumerate(content_parts):
                            message_html += f'<p>{part}</p>'
                            if i < len(image_data) and image_data[i] is not None:
                                img = display_image(image_data[i])
                                if img:
                                    # In a real implementation, embed the image directly
                                    message_html += f'<img src="data:image/png;base64,{image_to_base64(img)}" class="message-image" />'
                                else:
                                    message_html += f'<p><i>[Image unable to display]</i></p>'

                        message_html += '</div></div>'

                        st.session_state.conversation_html += message_html
                        conversation_area.markdown(st.session_state.conversation_html, unsafe_allow_html=True)

                    # Start collecting content for the new turn
                    current_turn_role = turn_start_match.group(1)
                    current_turn_content = turn_start_match.group(2)

                elif current_turn_role is not None:
                    # Check if this line is a log message by looking for common log patterns
                    log_patterns = ["INFO:", "DEBUG:", "WARNING:", "ERROR:", "CRITICAL:"]
                    is_log_line = any(line.strip().startswith(pattern) for pattern in log_patterns)

                    if not is_log_line:
                        # Append line to the current turn's content only if it's not a log line
                        is_conversation_line = True
                        current_turn_content += "\n" + line
                    else:
                        # This is a log line, so don't mark it as conversation line
                        is_conversation_line = False

                # Update logs (exclude conversation lines that will be shown in chat bubbles)
                if not is_conversation_line:
                    # Always update the log content, but control visibility with CSS
                    log_html = f'<div class="log-container" style="display:{"block" if st.session_state.show_logs else "none"}"><pre>{"".join(st.session_state.raw_log_lines)}</pre></div>'
                    log_placeholder.markdown(log_html, unsafe_allow_html=True)

            # After the loop, save and display the last turn if there is one
            if current_turn_role is not None and current_turn_content:
                # Map role names to standardized roles
                role_mapping = {
                    "USER": "human",
                    "HUMAN": "human",
                    "ASSISTANT": "assistant",
                    "AI": "assistant",
                    "SYSTEM": "system"
                }

                std_role = role_mapping.get(current_turn_role.upper(), "system")
                st.session_state.conversation_history.append({
                    "role": std_role,
                    "content": current_turn_content.strip()
                })

                # Process the content to extract thinking sections
                clean_content, thinking_sections = render_message_with_thinking(current_turn_content.strip())

                # Process content to find images
                content_parts, image_data = render_with_images(clean_content, image_files)

                # Create HTML for this message
                avatar_emoji = "🧑‍💻" if std_role == "human" else ("🤖" if std_role == "assistant" else "ℹ️")
                avatar_class = "human-avatar" if std_role == "human" else "assistant-avatar"
                bubble_class = "human-bubble" if std_role == "human" else "assistant-bubble"
                row_class = f"message-row {std_role}"

                message_html = ""

                # Add thinking sections BEFORE message bubble
                for thinking in thinking_sections:
                    # Clean up the thinking content
                    thinking = thinking.strip()
                    message_html += f'<div class="thinking-section"><strong>Thinking:</strong><pre>{thinking}</pre></div>'

                # Add the message bubble
                message_html += f'<div class="{row_class}">'
                message_html += f'<div class="avatar {avatar_class}">{avatar_emoji}</div>'
                message_html += f'<div class="{bubble_class}">'

                # Add content with images
                for i, part in enumerate(content_parts):
                    message_html += f'<p>{part}</p>'
                    if i < len(image_data) and image_data[i] is not None:
                        img = display_image(image_data[i])
                        if img:
                            # In a real implementation, embed the image directly
                            message_html += f'<img src="data:image/png;base64,{image_to_base64(img)}" class="message-image" />'
                        else:
                            message_html += f'<p><i>[Image unable to display]</i></p>'

                message_html += '</div></div>'

                st.session_state.conversation_html += message_html
                conversation_area.markdown(st.session_state.conversation_html, unsafe_allow_html=True)

            # Wait for the process to finish
            process.wait()

            if process.returncode != 0:
                st.error(f"Script failed with return code {process.returncode}")
            else:
                st.success("AI Battle completed.")

            # Check for disagreements
            st.session_state.has_disagreement = detect_disagreement(st.session_state.conversation_history)

            # Generate and display enhanced network graph
            if st.session_state.conversation_history:
                # Generate interactive network graph with drag-and-drop nodes
                flow_chart_html = '<div class="network-graph">'

                # Define node style based on role and disagreement
                node_classes = {
                    "human": "graph-node human",
                    "assistant": "graph-node assistant",
                    "system": "graph-node system"
                }

                # Add nodes for each message
                for i, turn in enumerate(st.session_state.conversation_history):
                    role = turn['role']

                    # Get a short preview of content
                    clean_content, _ = render_message_with_thinking(turn['content'])
                    content_preview = clean_content[:100].replace('"', "'") + "..." if len(clean_content) > 100 else clean_content.replace('"', "'")

                    # Check for disagreement
                    has_disagreement = False
                    if st.session_state.has_disagreement and i > 0:
                        disagreement_keywords = ["disagree", "incorrect", "not accurate", "wrong", "mistaken"]
                        has_disagreement = any(keyword in clean_content.lower() for keyword in disagreement_keywords)

                    # Create node class
                    node_class = node_classes.get(role, "graph-node")
                    if has_disagreement:
                        node_class += " disagreement"

                    # Create node with full content in tooltip
                    flow_chart_html += f'<div class="{node_class}" id="node-{i}">'
                    flow_chart_html += f'<h4>{role.upper()} {i+1}</h4>'
                    flow_chart_html += f'<p>{content_preview}</p>'

                    # Add tooltip with full content
                    flow_chart_html += f'<div class="node-tooltip">{clean_content}</div>'
                    flow_chart_html += '</div>'

                flow_chart_html += '</div>'

                # Create mermaid diagram first (for both tabs to use)
                # Compatible with Mermaid 11.6.0
                mermaid_syntax = "flowchart TD\n"

                # Define node style based on role and disagreement
                # Define class styles for Mermaid 11.6.0
                class_defs = """
    classDef human fill:#234b76,stroke:#1e88e5,stroke-width:2px,color:#fff;
    classDef assistant fill:#43a047,stroke:#7cb342,stroke-width:2px,color:#fff;
    classDef system fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;
    classDef default fill:#f5f5f5,stroke:#9e9e9e;
    classDef disagreement stroke-dasharray: 5 5;
"""

                # Add nodes for each message
                for i, turn in enumerate(st.session_state.conversation_history):
                    node_id = f"turn{i}"
                    role = turn['role']

                    # Get a short preview
                    clean_content, _ = render_message_with_thinking(turn['content'])
                    # Sanitize content for Mermaid - replace problematic characters
                    content_preview = clean_content[:40]
                    content_preview = (content_preview + "...") if len(clean_content) > 40 else content_preview
                    # Remove any characters that could break Mermaid syntax
                    content_preview = content_preview.replace('"', "'").replace('[', '(').replace(']', ')')
                    content_preview = content_preview.replace('\n', ' ').replace('\r', ' ')

                    # Add node with disagreement marker if needed
                    if st.session_state.has_disagreement and i > 0:
                        disagreement_keywords = ["disagree", "incorrect", "not accurate", "wrong", "mistaken"]
                        if any(keyword in clean_content.lower() for keyword in disagreement_keywords):
                            mermaid_syntax += f"    {node_id}[\"! {role.upper()}: {content_preview}\"]\n"
                            # Add class for disagreement
                            mermaid_syntax += f"    class {node_id} {role},disagreement;\n"
                        else:
                            mermaid_syntax += f"    {node_id}[\"{role.upper()}: {content_preview}\"]\n"
                            mermaid_syntax += f"    class {node_id} {role};\n"
                    else:
                        mermaid_syntax += f"    {node_id}[\"{role.upper()}: {content_preview}\"]\n"
                        mermaid_syntax += f"    class {node_id} {role};\n"

                # Add connections between nodes
                for i in range(1, len(st.session_state.conversation_history)):
                    prev_node_id = f"turn{i-1}"
                    node_id = f"turn{i}"

                    # Mark disagreement in the connections
                    if st.session_state.has_disagreement:
                        disagreement_keywords = ["disagree", "incorrect", "not accurate", "wrong", "mistaken"]
                        if any(keyword in st.session_state.conversation_history[i]['content'].lower() for keyword in disagreement_keywords):
                            mermaid_syntax += f"    {prev_node_id} -- \"Disagreement\" --> {node_id}\n"
                        else:
                            mermaid_syntax += f"    {prev_node_id} --> {node_id}\n"
                    else:
                        mermaid_syntax += f"    {prev_node_id} --> {node_id}\n"

                # Add class definitions at the end
                mermaid_syntax += class_defs

                # Show two tabs for viewing the graph
                graph_tab1, graph_tab2 = st.tabs(["Visual Graph", "Mermaid Diagram"])

                with graph_tab1:
                    st.markdown(flow_chart_html, unsafe_allow_html=True)

                with graph_tab2:
                    # Use HTML components to render mermaid properly
                    st.components.v1.html(
                        f"""
                        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/mermaid@11.6.0/dist/mermaid.min.css">
                        <script src="https://cdn.jsdelivr.net/npm/mermaid@11.6.0/dist/mermaid.min.js"></script>
                        <div class="mermaid">
                        {mermaid_syntax}
                        </div>
                        <script>
                            mermaid.initialize({{
                                startOnLoad: true,
                                theme: 'default',
                                flowchart: {{
                                    useMaxWidth: true,
                                    htmlLabels: true,
                                    curve: 'basis'
                                }}
                            }});
                        </script>
                        """,
                        height=400,
                    )
            else:
                flow_chart_area.info("No conversation history to generate flow chart.")

            # Show export button after completion
            export_placeholder = st.empty()
            export_placeholder.button("Export Results", key="export_after_completion", on_click=lambda: export_conversation(
                st.session_state.conversation_history,
                st.session_state.raw_log_lines,
                st.session_state.battle_id
            ))

        except FileNotFoundError:
            st.error(f"Error: Script not found at {RUN_SCRIPT_PATH}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.error(traceback.format_exc())
        finally:
            # Clean up the temporary config file
            if temp_config_path and os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
