#!/usr/bin/env python3
"""
Documentation generator for AI Battle framework.

This script extracts docstrings from the source code and generates markdown files
based on the templates. It then builds the documentation using MkDocs.
"""

import os
import re
import sys
import inspect
import importlib
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to document
try:
    import src
except ImportError:
    print("Error: Could not import src package. Make sure you're running this script from the project root.")
    sys.exit(1)

# Constants
TEMPLATE_DIR = Path('docs/api/templates')
OUTPUT_DIR = Path('docs/api')
SRC_DIR = Path('src')

# Templates
MODULE_TEMPLATE_PATH = TEMPLATE_DIR / 'module_template.md'
CLASS_TEMPLATE_PATH = TEMPLATE_DIR / 'class_template.md'
METHOD_TEMPLATE_PATH = TEMPLATE_DIR / 'method_template.md'


def load_template(template_path: Path) -> str:
    """Load a template from a file."""
    with open(template_path, 'r') as f:
        return f.read()


def save_markdown(content: str, output_path: Path) -> None:
    """Save markdown content to a file."""
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)


def extract_docstring(obj: Any) -> str:
    """Extract docstring from an object."""
    docstring = inspect.getdoc(obj) or ""
    return docstring


def get_constructor_signature(cls: type) -> str:
    """Get the constructor signature of a class."""
    try:
        signature = inspect.signature(cls.__init__)
        # Remove 'self' parameter
        params = list(signature.parameters.values())[1:]
        param_str = ', '.join(str(param) for param in params)
        return f"{cls.__name__}({param_str})"
    except (ValueError, TypeError):
        return f"{cls.__name__}()"


def get_method_signature(method: callable) -> str:
    """Get the signature of a method."""
    try:
        signature = inspect.signature(method)
        # Remove 'self' parameter if it's a method
        if 'self' in signature.parameters:
            params = list(signature.parameters.values())[1:]
            param_str = ', '.join(str(param) for param in params)
            return f"{method.__name__}({param_str})"
        else:
            return f"{method.__name__}{signature}"
    except (ValueError, TypeError):
        return f"{method.__name__}()"


def format_parameters(obj: callable) -> str:
    """Format parameters for documentation."""
    try:
        signature = inspect.signature(obj)
        params = []
        for name, param in signature.parameters.items():
            if name == 'self':
                continue
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else 'Any'
            param_default = f" (default: {param.default})" if param.default != inspect.Parameter.empty else ""
            params.append(f"- **{name}** (`{param_type}`): {param_default}")
        return '\n'.join(params)
    except (ValueError, TypeError):
        return "No parameters."


def format_return_value(obj: callable) -> str:
    """Format return value for documentation."""
    try:
        signature = inspect.signature(obj)
        return_type = signature.return_annotation
        if return_type == inspect.Signature.empty:
            return "None"
        return f"`{return_type}`"
    except (ValueError, TypeError):
        return "Unknown"


def generate_module_doc(module_name: str, module: Any) -> None:
    """Generate documentation for a module."""
    print(f"Generating documentation for module: {module_name}")
    
    # Load template
    template = load_template(MODULE_TEMPLATE_PATH)
    
    # Extract module information
    module_description = extract_docstring(module)
    
    # Get classes and functions
    classes = []
    functions = []
    for name, obj in inspect.getmembers(module):
        if name.startswith('_'):
            continue
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            classes.append(f"- [{name}]({name.lower()}.md): {extract_docstring(obj).split('.')[0]}")
            generate_class_doc(name, obj, module_name)
        elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
            functions.append(f"- [{name}]({name.lower()}.md): {extract_docstring(obj).split('.')[0]}")
            generate_method_doc(name, obj, module_name)
    
    # Fill template
    content = template.format(
        module_name=module_name,
        module_description=module_description,
        class_list='\n'.join(classes) if classes else "No classes in this module.",
        function_list='\n'.join(functions) if functions else "No functions in this module.",
        usage_example="# TODO: Add usage example",
        related_modules="# TODO: Add related modules"
    )
    
    # Save markdown
    output_path = OUTPUT_DIR / module_name.lower().replace('.', '/') / 'index.md'
    save_markdown(content, output_path)


def generate_class_doc(class_name: str, cls: type, module_name: str) -> None:
    """Generate documentation for a class."""
    print(f"  Generating documentation for class: {class_name}")
    
    # Load template
    template = load_template(CLASS_TEMPLATE_PATH)
    
    # Extract class information
    class_description = extract_docstring(cls)
    constructor_signature = get_constructor_signature(cls)
    constructor_parameters = format_parameters(cls.__init__)
    
    # Get properties and methods
    properties = []
    methods = []
    for name, obj in inspect.getmembers(cls):
        if name.startswith('_') and name != '__init__':
            continue
        if isinstance(obj, property):
            properties.append(f"- **{name}**: {extract_docstring(obj).split('.')[0]}")
        elif inspect.isfunction(obj) or inspect.ismethod(obj):
            if name != '__init__':
                methods.append(f"- [{name}]({class_name.lower()}_{name.lower()}.md): {extract_docstring(obj).split('.')[0]}")
                generate_method_doc(f"{class_name}_{name}", obj, module_name, class_name)
    
    # Fill template
    content = template.format(
        class_name=class_name,
        class_description=class_description,
        constructor_signature=constructor_signature,
        constructor_parameters=constructor_parameters,
        properties_list='\n'.join(properties) if properties else "No properties.",
        methods_list='\n'.join(methods) if methods else "No methods.",
        usage_example="# TODO: Add usage example",
        related_classes="# TODO: Add related classes"
    )
    
    # Save markdown
    output_path = OUTPUT_DIR / module_name.lower().replace('.', '/') / f"{class_name.lower()}.md"
    save_markdown(content, output_path)


def generate_method_doc(method_name: str, method: callable, module_name: str, class_name: Optional[str] = None) -> None:
    """Generate documentation for a method or function."""
    print(f"    Generating documentation for method: {method_name}")
    
    # Load template
    template = load_template(METHOD_TEMPLATE_PATH)
    
    # Extract method information
    method_description = extract_docstring(method)
    method_signature = get_method_signature(method)
    parameters = format_parameters(method)
    return_value = format_return_value(method)
    
    # Fill template
    content = template.format(
        method_name=method_name,
        method_signature=method_signature,
        method_description=method_description,
        parameters_list=parameters,
        return_value=return_value,
        exceptions_list="# TODO: Add exceptions",
        usage_example="# TODO: Add usage example",
        notes="# TODO: Add notes",
        see_also="# TODO: Add see also"
    )
    
    # Save markdown
    if class_name:
        output_path = OUTPUT_DIR / module_name.lower().replace('.', '/') / f"{method_name.lower()}.md"
    else:
        output_path = OUTPUT_DIR / module_name.lower().replace('.', '/') / f"{method_name.lower()}.md"
    save_markdown(content, output_path)


def discover_modules() -> List[Tuple[str, Any]]:
    """Discover modules to document."""
    modules = []
    
    # Check if src directory exists
    if not SRC_DIR.exists():
        print(f"Error: {SRC_DIR} directory not found.")
        return modules
    
    # Discover modules
    for path in SRC_DIR.glob('**/*.py'):
        if path.name.startswith('_'):
            continue
        
        # Convert path to module name
        relative_path = path.relative_to(SRC_DIR.parent)
        module_name = str(relative_path).replace('/', '.').replace('\\', '.')[:-3]
        
        try:
            module = importlib.import_module(module_name)
            modules.append((module_name, module))
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")
    
    return modules


def main() -> None:
    """Main function."""
    print("Generating API documentation...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Discover modules
    modules = discover_modules()
    
    # Generate documentation for each module
    for module_name, module in modules:
        generate_module_doc(module_name, module)
    
    print("Documentation generation complete.")


if __name__ == '__main__':
    main()