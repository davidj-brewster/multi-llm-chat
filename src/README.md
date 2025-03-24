# AI Battle Framework Source Code

This directory contains the source code for the AI Battle framework, organized into logical modules.

## Directory Structure

- **core/**: Core components of the framework
  - ConversationManager
  - Adaptive Instructions

- **configuration/**: Configuration system
  - Configuration Classes
  - YAML Integration

- **model_clients/**: Model client implementations
  - BaseClient
  - Model-Specific Clients (OpenAI, Claude, Gemini, etc.)

- **file_handling/**: File handling components
  - FileMetadata
  - ConversationMediaHandler

- **metrics/**: Metrics and analysis components
  - MetricsAnalyzer
  - TopicAnalyzer

- **arbiter/**: Arbiter system components
  - ConversationArbiter
  - AssertionGrounder

- **utilities/**: Utility functions and classes
  - MemoryManager
  - Helper Functions

## Code Organization

The code is organized following these principles:

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Dependency Injection**: Dependencies are injected rather than created internally
3. **Interface Segregation**: Interfaces are kept small and focused
4. **Open/Closed Principle**: Code is open for extension but closed for modification

## Development Guidelines

When contributing to the codebase, please follow these guidelines:

1. **Documentation**: All classes and methods should be documented with docstrings
2. **Type Annotations**: Use type annotations for all function parameters and return values
3. **Error Handling**: Use appropriate error handling and provide meaningful error messages
4. **Testing**: Write tests for all new functionality
5. **Code Style**: Follow PEP 8 guidelines for Python code style

## Building and Testing

To build and test the framework, use the following commands:

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
```

## API Documentation

For detailed API documentation, see the [API Documentation](../docs/api/index.md).