#!/usr/bin/env python3
"""
Migration script for AI Battle framework.

This script helps with the migration of the existing code to the new src directory structure.
It creates the necessary directories, moves the existing Python files to the appropriate
directories, and updates imports in the moved files.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# File mapping: source file -> destination directory
FILE_MAPPING = {
    'ai-battle.py': 'src/core',
    'adaptive_instructions.py': 'src/core',
    'configuration.py': 'src/configuration',
    'configdataclasses.py': 'src/configuration',
    'config_integration.py': 'src/configuration',
    'model_clients.py': 'src/model_clients',
    'file_handler.py': 'src/file_handling',
    'metrics_analyzer.py': 'src/metrics',
    'arbiter_v4.py': 'src/arbiter',
    'shared_resources.py': 'src/utilities',
    'context_analysis.py': 'src/metrics',
}

# Create __init__.py files in these directories
INIT_DIRS = [
    'src',
    'src/core',
    'src/configuration',
    'src/model_clients',
    'src/file_handling',
    'src/metrics',
    'src/arbiter',
    'src/utilities',
]

# Rename files during migration
RENAME_FILES = {
    'ai-battle.py': 'conversation_manager.py',
    'arbiter_v4.py': 'arbiter.py',
}

# Import patterns to update
IMPORT_PATTERNS = [
    (r'from (adaptive_instructions|configuration|configdataclasses|config_integration|model_clients|file_handler|metrics_analyzer|arbiter_v4|shared_resources|context_analysis) import', r'from src.\1 import'),
    (r'import (adaptive_instructions|configuration|configdataclasses|config_integration|model_clients|file_handler|metrics_analyzer|arbiter_v4|shared_resources|context_analysis)', r'import src.\1'),
]


def create_directories() -> None:
    """Create the necessary directories in the src directory."""
    print("Creating directories...")
    for directory in set(FILE_MAPPING.values()):
        os.makedirs(directory, exist_ok=True)

    for directory in INIT_DIRS:
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f'"""{os.path.basename(directory)} module for AI Battle framework."""\n')
            print(f"Created {init_file}")


def move_files() -> None:
    """Move the existing Python files to the appropriate directories."""
    print("Moving files...")
    for source_file, dest_dir in FILE_MAPPING.items():
        if not os.path.exists(source_file):
            print(f"Warning: {source_file} not found, skipping.")
            continue

        # Get the destination filename (possibly renamed)
        dest_filename = RENAME_FILES.get(source_file, source_file)
        dest_filename = os.path.basename(dest_filename)
        dest_path = os.path.join(dest_dir, dest_filename)

        # Create a backup of the original file
        backup_path = f"{source_file}.bak"
        shutil.copy2(source_file, backup_path)
        print(f"Created backup: {backup_path}")

        # Copy the file to the destination
        shutil.copy2(source_file, dest_path)
        print(f"Moved {source_file} to {dest_path}")


def update_imports() -> None:
    """Update imports in the moved files."""
    print("Updating imports...")
    for _, dest_dir in FILE_MAPPING.items():
        for root, _, files in os.walk(dest_dir):
            for file in files:
                if not file.endswith('.py'):
                    continue

                file_path = os.path.join(root, file)
                update_file_imports(file_path)


def update_file_imports(file_path: str) -> None:
    """Update imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()

    original_content = content
    for pattern, replacement in IMPORT_PATTERNS:
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated imports in {file_path}")


def main() -> None:
    """Main function."""
    print("Starting migration...")

    # Create directories
    create_directories()

    # Move files
    move_files()

    # Update imports
    update_imports()

    print("Migration complete.")
    print("Note: The original files have been backed up with .bak extension.")
    print("You can delete them after verifying that everything works correctly.")


if __name__ == '__main__':
    main()
