# AI Battle Framework API Documentation

Welcome to the comprehensive API documentation for the AI Battle framework. This documentation provides detailed information about all classes, methods, and modules in the framework.

## Overview

The AI Battle framework is a sophisticated system for orchestrating conversations between multiple AI models, with features for file handling, configuration management, metrics analysis, and conversation evaluation.

## Core Components

- [ConversationManager](core/conversation_manager.md): The main class that manages conversations between AI models
- [Adaptive Instructions](core/adaptive_instructions.md): System for dynamically generating instructions

## Configuration System

- [Configuration Classes](configuration/config_classes.md): Data classes for configuration
- [YAML Integration](configuration/yaml_integration.md): Loading and validating YAML configurations

## Model Clients

- [BaseClient](model_clients/base_client.md): Base class for all model clients
- [Model-Specific Clients](model_clients/specific_clients.md): Implementations for different AI models

## File Handling

- [FileMetadata](file_handling/file_metadata.md): Data class for file metadata
- [ConversationMediaHandler](file_handling/media_handler.md): Handler for processing media files

## Metrics & Analysis

- [MetricsAnalyzer](metrics/metrics_analyzer.md): Analyzer for conversation metrics
- [TopicAnalyzer](metrics/topic_analyzer.md): Analyzer for conversation topics

## Arbiter System

- [ConversationArbiter](arbiter/conversation_arbiter.md): Evaluator for conversations
- [AssertionGrounder](arbiter/assertion_grounder.md): Grounder for assertions

## Utilities

- [MemoryManager](utilities/memory_manager.md): Manager for memory optimization
- [Helper Functions](utilities/helper_functions.md): Utility functions

## Usage Examples

- [Basic Usage](examples/basic_usage.md): Basic usage examples
- [Configuration-Driven](examples/configuration_driven.md): Configuration-driven examples
- [File-Based Discussions](examples/file_based.md): File-based discussion examples
- [Metrics Collection](examples/metrics_collection.md): Metrics collection examples