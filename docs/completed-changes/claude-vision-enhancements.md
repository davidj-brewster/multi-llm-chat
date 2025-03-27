# Claude Vision and Medical Imagery Enhancements

## Overview

This document outlines the significant enhancements made to the Claude client implementation, focusing on advanced vision capabilities, medical imagery analysis, and video frame processing.

## Major Enhancements

### 1. Advanced Medical Imagery Analysis

The Claude client has been upgraded to better handle medical images with specialized capabilities:

- **High-Resolution Support**: Increased maximum resolution to 1800px (up from 1024px)
- **Enhanced Detail Detection**: Special metadata flags for medical images to trigger detailed analysis
- **Medical-Specific Instruction Sets**: Automatic addition of medical analysis instructions when medical images are detected
- **Signal Intensity Analysis**: Improved detection of hyperintensities and subtle tissue variations
- **Tissue Differentiation**: Better identification of anatomical structures in medical scans

### 2. Multi-Image Processing

- **Support for up to 10 images per request**: Analyze multiple medical images in a single API call
- **Comparative Analysis**: Framework for comparing findings across multiple images
- **Sequential Processing**: Logical ordering of multiple images for progressive analysis
- **Metadata Enrichment**: Each image can have specific metadata for enhanced analysis

### 3. Video Frame Analysis (New!)

- **Key Frame Extraction**: Support for analyzing video content through key frames
- **Temporal Context**: Metadata includes timestamps and frame numbers for temporal analysis
- **Frame Sequencing**: Frames are presented in sequence with context for time-based patterns
- **Motion Analysis**: Specialized prompting for analyzing changes between frames
- **Video-Specific Instructions**: Automatic addition of video analysis guidance

### 4. Claude 3.7 Integration

- **Updated Default Model**: Now using "claude-3-7-sonnet" as the default model
- **Reasoning Parameter Support**: Added support for controlling Claude's reasoning level
- **Capability Detection**: Automatic detection of model capabilities based on version
- **Advanced Parameter Control**: Support for seed values and stop sequences

### 5. System Prompting Enhancements

Medical imagery system prompts now include specialized instructions:
- Pay attention to subtle variations in tissue density and signal intensity
- Identify anatomical structures with precision
- Note asymmetries, hyperintensities, or abnormal patterns
- Compare across multiple images to identify patterns
- Special attention to signal changes in T1/T2/FLAIR weighted sequences
- Track changes across video frames for temporal analysis

## Implementation Details

### Key Configuration Parameters

```python
# Enhanced vision parameters
self.vision_max_resolution = 1800  # Up from default 1024
self.max_images_per_request = 10   # Support for multiple images
self.high_detail_vision = True     # Enable detailed medical image analysis
self.video_frame_support = True    # Enable video frame extraction

# Reasoning parameters
self.reasoning_level = "auto"      # Options: none, low, medium, high, auto
```

### Capability Detection

The client now automatically detects and enables features based on the Claude model:

```python
# All Claude 3 models support vision
self.capabilities["vision"] = True

# Claude 3.5 and newer models support video frames
if any(m in self.model.lower() for m in ["claude-3.5", "claude-3-5", "claude-3-7"]):
    self.capabilities["video_frames"] = True
    self.capabilities["high_resolution"] = True
    self.capabilities["medical_imagery"] = True

# Claude 3.7 and newer models support advanced reasoning
if "claude-3-7" in self.model.lower():
    self.capabilities["advanced_reasoning"] = True
```

### Medical Image Metadata

When processing medical images, special metadata is added:

```python
image_metadata = {}
                
# Add high resolution support for medical images
if self.capabilities.get("high_resolution", False):
    # Add high resolution flag for detailed medical imagery
    image_metadata["high_resolution"] = True
    
    # Add detailed medical image analysis flag if applicable
    if self.capabilities.get("medical_imagery", False) and "medical" in file_item.get("path", "").lower():
        image_metadata["detail"] = "high"
```

### Video Frame Processing

The client now supports video analysis through frame extraction:

```python
# Process video frames
elif file_item["type"] == "video" and self.capabilities.get("video_frames", False):
    # Extract key frames if available
    if "key_frames" in file_item and file_item["key_frames"]:
        for frame in file_item["key_frames"][:self.max_images_per_request - image_count]:
            video_frames.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame["base64"]
                },
                "metadata": {
                    "timestamp": frame.get("timestamp", 0),
                    "frame_number": frame.get("frame_number", 0),
                    "video_source": file_item.get("path", "unknown")
                }
            })
```

## Usage Examples

### Analyzing Medical MRI Images

```python
client = ClaudeClient(
    role="assistant",
    api_key=api_key,
    mode="default",
    domain="Medical Imaging",
    model="claude-3-7-sonnet"
)

response = client.generate_response(
    prompt="Please analyze this brain MRI scan for any abnormalities, focusing on potential hyperintensities or structural changes",
    file_data=mri_image_data,
    model_config=ModelConfig(temperature=0.2)  # Lower temperature for medical analysis
)
```

### Processing Video Frames

```python
client = ClaudeClient(
    role="assistant",
    api_key=api_key,
    mode="default",
    domain="Medical Imaging",
    model="claude-3-7-sonnet"
)

response = client.generate_response(
    prompt="Analyze these frames from an MRI video sequence, noting any dynamic changes or patterns between frames",
    file_data=video_data,  # Contains key_frames
    model_config=ModelConfig(temperature=0.3)
)
```

## Future Improvements

1. **Further Resolution Optimization**: Implement dynamic resolution scaling based on image content
2. **Specialized Medical Models**: Add support for domain-specific Claude versions when available
3. **Automatic Frame Selection**: Smart selection of the most relevant video frames
4. **Multimodal Cross-Reference**: Link textual medical reports with imagery for comprehensive analysis
5. **Region-of-Interest Support**: Add capability to highlight and analyze specific image regions