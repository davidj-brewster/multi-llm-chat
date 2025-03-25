# Multi-File Vision Support Implementation Plan

## Problem Analysis

After thoroughly analyzing the codebase, I've identified the key issue preventing multiple files from being properly processed and sent to the models:

1. In `ai-battle.py`, the `run_conversation_with_file` method processes multiple files correctly but then structures the data incorrectly:
   ```python
   # Current problematic code
   if file_data_list:
       file_data = file_data_list[0]  # Takes only the first file as main file_data
       if len(file_data_list) > 1:
           file_data["additional_files"] = file_data_list[1:]  # Adds rest as a property
   ```

2. However, in `model_clients.py`, the `generate_response` method expects a different format for multiple files:
   ```python
   # How the client expects multiple files
   if isinstance(file_data, list) and file_data:
       # Handle multiple files as a list
       # ...
   ```

3. This mismatch means that when multiple files are provided, the model only receives the first file with the others stored in an "additional_files" property that is never processed.

## Solution Approach

The solution is to modify how file data is passed between components to ensure consistent handling of multiple files:

1. Update `run_conversation_with_file` in `ai-battle.py` to pass the entire list of file data to the model client when multiple files are present.

2. Ensure the `_run_conversation_with_file_data` method correctly passes this list to `run_conversation_turn`.

3. Verify that all model clients can properly handle a list of file data.

## Implementation Details

### 1. Fix in `ai-battle.py`

The key change is in the `run_conversation_with_file` method:

```python
# For MultiFileConfig case (around line 403)
if file_data_list:
    # Instead of taking just the first file, pass the entire list
    file_data = file_data_list  # Pass the entire list
    
# For dictionary format case (around line 462)
if file_data_list:
    # Instead of taking just the first file, pass the entire list
    file_data = file_data_list  # Pass the entire list
```

### 2. Update Ollama Client for Multiple Files

The Ollama client needs to be updated to handle multiple files:

```python
# In model_clients.py (around line 1724)
if is_vision_model and file_data:
    if isinstance(file_data, list) and file_data:
        # Handle multiple files
        all_images = []
        for file_item in file_data:
            if file_item["type"] == "image" and "base64" in file_item:
                all_images.append(file_item["base64"])
            elif file_item["type"] == "video" and "key_frames" in file_item and file_item["key_frames"]:
                all_images.extend([frame["base64"] for frame in file_item["key_frames"]])
        
        if all_images:
            messages[-1]['images'] = all_images
    elif file_data["type"] == "image" and "base64" in file_data:
        # Original single image handling
        messages[-1]['images'] = [file_data['base64']]
    # Rest of the code for video handling...
```

### 3. Model Client Support Analysis

After reviewing all model clients, here's the status of multiple file support:

1. **Gemini Client**: Already has proper support for multiple files (lines 947-978)
   ```python
   if isinstance(file_data, list) and file_data:
       # Handle multiple files
       image_parts = []
       text_content = ""
       # Process all files...
   ```

2. **Claude Client**: Already has proper support for multiple files (lines 1190-1234)
   ```python
   if isinstance(file_data, list) and file_data:
       # Handle multiple files
       if self.capabilities.get("vision", False):
           # Format for Claude's multimodal API with multiple images
           message_content = []
           # Process all files...
   ```

3. **OpenAI Client**: Already has proper support for multiple files (lines 1365-1402)
   ```python
   if isinstance(file_data, list) and file_data:
       # Handle multiple files
       if self.capabilities.get("vision", False):
           # Format for OpenAI's vision API with multiple images
           content_parts = [{"type": "text", "text": prompt}]
           # Process all files...
   ```

4. **Ollama Client**: Needs to be updated to handle multiple files (as shown above)

## Testing Strategy

1. Test with a single file via command line:
   ```
   python examples/run_vision_discussion.py examples/configs/vision_discussion.yaml ./T2FLAIR-SPACE-SAG-CS.mov
   ```

2. Test with multiple files via command line:
   ```
   python examples/run_vision_discussion.py examples/configs/vision_discussion.yaml ./T2-SAG-FLAIR.mov ./T2FLAIR-SPACE-SAG-CS.mov
   ```

3. Test with a configuration file that specifies multiple files:
   ```
   python examples/run_vision_discussion.py examples/configs/multi_file_vision_discussion.yaml
   ```

## Expected Outcome

After these changes:

1. Models will receive all files when multiple files are provided
2. The conversation will include references to all files
3. The HTML output will display all files correctly

This implementation maintains backward compatibility with single files while properly supporting multiple files.