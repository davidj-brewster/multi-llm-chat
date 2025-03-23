# AI Battle Framework Documentation

This directory contains the documentation for the AI Battle framework.

## Documentation Structure

- **api/**: API documentation for all classes and methods
- **requirements.txt**: Dependencies for documentation generation
- **generate_docs.py**: Script to generate API documentation from source code
- **setup_docs.sh**: Script to set up the documentation environment

## Setting Up the Documentation Environment

### Manual Setup

1. Create a virtual environment:
   ```bash
   python -m venv docs_venv
   ```

2. Activate the virtual environment:
   ```bash
   # On Linux/macOS
   source docs_venv/bin/activate
   
   # On Windows
   docs_venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r docs/requirements.txt
   ```

### Using the Setup Script

Alternatively, you can use the setup script:

```bash
# Make the script executable
chmod +x docs/setup_docs.sh

# Run the script
./docs/setup_docs.sh
```

## Generating Documentation

To generate the API documentation:

```bash
python docs/generate_docs.py
```

## Building Documentation with MkDocs

To build the documentation with MkDocs:

```bash
# Build the documentation
mkdocs build

# Serve the documentation locally
mkdocs serve
```

Then open http://localhost:8000 in your browser.

## Documentation Maintenance

When updating the documentation:

1. Update the source code docstrings
2. Run the documentation generator
3. Build the documentation with MkDocs
4. Review the documentation for accuracy and completeness

## Documentation Format

The documentation follows a consistent format:

- **Module Documentation**: Overview of the module's purpose and functionality
- **Class Documentation**: Detailed documentation of classes, including constructor parameters, properties, and methods
- **Method Documentation**: Detailed documentation of methods, including parameters, return values, and exceptions

## Contributing to Documentation

When contributing to the documentation, please follow these guidelines:

1. Use the provided templates for module, class, and method documentation
2. Include practical examples for each component
3. Ensure cross-references are correct
4. Keep the documentation in sync with the code