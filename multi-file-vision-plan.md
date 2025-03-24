# Multi-File Vision Support Implementation Plan

This document outlines the architecture and implementation plan for adding support for multiple input files to the vision discussion feature, particularly focusing on handling multiple image files across different model clients (OpenAI, Claude, Gemini, and Ollama).

## Overview

The current implementation supports a single input file (image, video, or text) for vision discussions. This enhancement will extend the system to support multiple input files, allowing for more comprehensive analyses and comparisons between different images or videos.

## Implementation Phases

### Phase 1: Core Data Structures and File Handling (Completed)

1. **Update Configuration Classes**
   - Added `is_directory` and `file_pattern` fields to `FileConfig` class
   - Created new `MultiFileConfig` class to handle multiple files or directories
   - Updated `DiscussionConfig` class to include `input_files` field

2. **Update File Handler**
   - Added `process_directory` method to scan directories for files matching a pattern
   - Added `process_multiple_files` method with robust error handling
   - Added `prepare_multiple_media_messages` method to prepare multiple files for conversation

### Phase 2: Model Client Updates (Completed)

1. **Base Client Updates**
   - Added `_prepare_multiple_file_content` method to `BaseClient` class

2. **Model-Specific Implementations**
   - **OllamaClient**: Updated to handle multiple text/code files by combining content
   - **ClaudeClient**: Updated to support multiple images in a single message
   - **OpenAIClient**: Updated to support multiple images in a single message
   - **GeminiClient**: Updated to handle multiple images as separate messages

### Phase 3: Run Script and Configuration Updates (Completed)

1. **Updated Run Script**
   - Modified `examples/run_vision_discussion.py` to accept multiple file paths
   - Added support for processing multiple files and directories
   - Updated file saving logic to handle multiple files

2. **Created Sample Configuration**
   - Created `examples/configs/multi_file_vision_discussion.yaml` as a template for multi-file discussions

## Architecture Details

### Data Flow

1. **Configuration Loading**
   - Load configuration from YAML file or command-line arguments
   - Parse `input_files` section if present, or fall back to `input_file` for backward compatibility

2. **File Processing**
   - Process individual files or scan directories based on configuration
   - Generate metadata for each file (type, dimensions, etc.)
   - Create `FileConfig` objects for each file

3. **Conversation Preparation**
   - Prepare appropriate prompts based on file types
   - Create messages with file content for model consumption

4. **Model Handling**
   - Each model client handles multiple files according to its API requirements
   - Some models support multiple images in a single request, others require separate messages

5. **Response Processing**
   - Process model responses and continue the conversation
   - Save conversation with all file data for HTML output

### Key Components

1. **MultiFileConfig**
   - Manages a collection of files or a directory of files
   - Provides options for filtering files by pattern
   - Limits the number of files to prevent performance issues

2. **File Handler Extensions**
   - Processes multiple files efficiently
   - Handles errors gracefully, continuing with valid files even if some fail
   - Prepares file content in the format required by each model

3. **Model Client Adaptations**
   - Each model client implements multi-file support according to its API capabilities
   - Maintains backward compatibility with single-file workflows

## Usage Examples

### Command Line

```bash
# Run with multiple individual files
python examples/run_vision_discussion.py examples/configs/multi_file_vision_discussion.yaml file1.jpg file2.jpg

# Run with configuration file that specifies multiple files
python examples/run_vision_discussion.py examples/configs/multi_file_vision_discussion.yaml
```

### Configuration File

```yaml
discussion:
  # ... other configuration ...
  
  # Option 1: List individual files
  input_files:
    files:
      - path: "image1.jpg"
        type: "image"
      - path: "image2.jpg"
        type: "image"
    
  # Option 2: Specify a directory
  input_files:
    directory: "path/to/image/directory"
    file_pattern: "*.jpg"
    max_files: 10
```

## Testing Strategy

1. **Unit Tests**
   - Test `MultiFileConfig` validation
   - Test directory scanning and file filtering
   - Test error handling for invalid files

2. **Integration Tests**
   - Test end-to-end workflow with multiple files
   - Test with different model combinations
   - Test with different file types (images, videos, text)

3. **Manual Testing**
   - Verify HTML output with multiple files
   - Check model responses for multi-file awareness
   - Validate performance with varying numbers of files

## Future Enhancements

1. **Advanced File Grouping**
   - Group related files for more contextual analysis
   - Support for file sequences or time series

2. **Improved File Visualization**
   - Enhanced HTML output for multiple files
   - Interactive comparison views

3. **Selective File Processing**
   - Allow models to request specific files for deeper analysis
   - Support for progressive loading of files during conversation

## Implementation Notes

- The implementation maintains backward compatibility with existing single-file workflows
- Error handling is robust, allowing the system to continue even if some files fail to process
- The design is flexible, supporting both explicit file lists and directory scanning
- Model-specific adaptations ensure optimal handling of multiple files according to each API's capabilities

## Implementation Summary

# Multi-File Vision Support Implementation

I've successfully implemented support for multiple input files in the vision discussion feature, allowing the system to handle multiple images or videos across different model clients (OpenAI, Claude, Gemini, and Ollama).

## Completed Implementation

The implementation follows a three-phase approach as outlined in the `multi-file-vision-plan.md` document:

### Phase 1: Core Data Structures and File Handling
- Updated `FileConfig` class with `is_directory` and `file_pattern` fields
- Created new `MultiFileConfig` class to handle multiple files or directories
- Updated `DiscussionConfig` class to include `input_files` field
- Enhanced `file_handler.py` with methods for processing directories and multiple files

### Phase 2: Model Client Updates
- Added `_prepare_multiple_file_content` method to `BaseClient` class
- Updated all model clients (Ollama, Claude, OpenAI, Gemini) to handle multiple files according to their specific API requirements

### Phase 3: Run Script and Configuration Updates
- Modified `examples/run_vision_discussion.py` to accept multiple file paths
- Added support for processing multiple files and directories
- Updated file saving logic to handle multiple files
- Created `examples/configs/multi_file_vision_discussion.yaml` as a template

## Key Features

1. **Multiple File Support**: Process multiple images or videos in a single conversation
2. **Directory Scanning**: Automatically scan directories for files matching a pattern
3. **Flexible Configuration**: Configure via YAML or command-line arguments
4. **Backward Compatibility**: Maintains support for existing single-file workflows
5. **Model-Specific Adaptations**: Each model client handles multiple files according to its API capabilities

## Usage Instructions

### Command Line
```bash
# Run with multiple individual files
python examples/run_vision_discussion.py examples/configs/multi_file_vision_discussion.yaml file1.jpg file2.jpg

# Run with configuration file that specifies multiple files
python examples/run_vision_discussion.py examples/configs/multi_file_vision_discussion.yaml
```

### Configuration File
The new `multi_file_vision_discussion.yaml` demonstrates two approaches:

1. **Explicit file list**:
```yaml
input_files:
  files:
    - path: "T2-SAG-FLAIR.mov"
      type: "video"
    - path: "T2FLAIR-SPACE-SAG-CS.mov"
      type: "video"
```

2. **Directory scanning** (commented out in the example):
```yaml
input_files:
  directory: "path/to/image/directory"
  file_pattern: "*.jpg"
  max_files: 10
```

## Implementation Details

The implementation maintains backward compatibility while adding robust support for multiple files. Each model client handles multiple files according to its API capabilities:

- **Ollama**: Combines text content from multiple files
- **Claude**: Supports multiple images in a single message
- **OpenAI**: Supports multiple images in a single message
- **Gemini**: Handles multiple images as separate messages

The system includes robust error handling, continuing with valid files even if some fail to process.