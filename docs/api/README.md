# AI Battle Framework API Documentation

This directory contains comprehensive API documentation for the AI Battle framework.

## Documentation Structure

The documentation is organized into the following sections:

- **core/**: Core components of the framework (ConversationManager, Adaptive Instructions)
- **configuration/**: Configuration system (Configuration Classes, YAML Integration)
- **model_clients/**: Model client implementations (BaseClient, Model-Specific Clients)
- **file_handling/**: File handling components (FileMetadata, ConversationMediaHandler)
- **metrics/**: Metrics and analysis components (MetricsAnalyzer, TopicAnalyzer)
- **arbiter/**: Arbiter system components (ConversationArbiter, AssertionGrounder)
- **utilities/**: Utility functions and classes (MemoryManager, Helper Functions)
- **templates/**: Documentation templates for modules, classes, and methods

## Documentation Format

Each component is documented using a consistent format:

1. **Module Documentation**: Overview of the module's purpose and functionality
2. **Class Documentation**: Detailed documentation of classes, including constructor parameters, properties, and methods
3. **Method Documentation**: Detailed documentation of methods, including parameters, return values, and exceptions

## Using the Documentation

The documentation can be accessed in several ways:

1. **Markdown Files**: Browse the documentation directly in the repository
2. **HTML Documentation**: Generate HTML documentation using a documentation generator
3. **PDF Documentation**: Generate PDF documentation for offline reference

## Generating Documentation

To generate HTML documentation from the markdown files, you can use a documentation generator like MkDocs:

```bash
# Install MkDocs
pip install mkdocs

# Generate HTML documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Contributing to Documentation

When contributing to the documentation, please follow these guidelines:

1. Use the provided templates for module, class, and method documentation
2. Include practical examples for each component
3. Ensure cross-references are correct
4. Keep the documentation in sync with the code

## Documentation Maintenance

The documentation should be updated whenever the code changes. This includes:

1. Adding documentation for new components
2. Updating documentation for modified components
3. Removing documentation for deleted components
4. Updating examples to reflect current usage patterns