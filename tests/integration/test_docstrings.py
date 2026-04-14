"""Unified script to check docstrings across the entire papylio project.

Provides a single comprehensive tool to verify that all Python files have
proper module docstrings, class docstrings, and method/function docstrings.
"""
import ast
from pathlib import Path
import pytest
from collections import defaultdict
from typing import Dict, List, Tuple, Set


def check_docstrings(
    target_paths: List[str] = None,
    exclude_dirs: Set[str] = None,
    check_methods: bool = True,
    check_functions: bool = True,
    description: str = "Checking docstrings"
) -> Tuple[bool, Dict]:
    """Check docstrings in specified paths.

    Parameters
    ----------
    target_paths : List[str], optional
        List of paths to check (glob patterns supported). Defaults to all papylio files.
    exclude_dirs : Set[str], optional
        Directories to exclude from checks. Defaults to common non-source directories.
    check_methods : bool, optional
        Whether to check class methods for docstrings. Default True.
    check_functions : bool, optional
        Whether to check top-level functions for docstrings. Default True.
    description : str, optional
        Description of this check for output. Default "Checking docstrings".

    Returns
    -------
    Tuple[bool, Dict]
        (success, results) where success indicates all checks passed,
        and results contains detailed findings.
    """
    if exclude_dirs is None:
        exclude_dirs = {'__pycache__', 'docs', 'build', 'tests'}

    if target_paths is None:
        target_paths = ['papylio']

    print("=" * 70)
    print(description.center(70))
    print("=" * 70)

    results = defaultdict(lambda: {'module': False, 'classes': [], 'functions': [], 'methods': []})
    errors = []

    # Collect all files matching target paths
    all_files = []
    for target in target_paths:
        target_path = Path(target)
        if target_path.is_file():
            all_files.append(target_path)
        else:
            all_files.extend(sorted(target_path.rglob('*.py')))

    # Process each file
    for file_path in all_files:
        # Skip excluded directories
        if any(excluded in file_path.parts for excluded in exclude_dirs):
            continue

        relative_path = str(file_path.relative_to(file_path.parts[0]))

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            # Check module docstring
            module_doc = ast.get_docstring(tree)
            results[relative_path]['module'] = bool(module_doc)

            # Check classes and their methods
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    doc = ast.get_docstring(node)
                    if not doc:
                        results[relative_path]['classes'].append(node.name)

                    # Check methods within classes if requested
                    if check_methods:
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                # Only require docstrings for public methods (no leading underscore)
                                if item.name.startswith('_'):
                                    continue
                                method_doc = ast.get_docstring(item)
                                if not method_doc and item.name not in ('__init__',):
                                    results[relative_path]['methods'].append(f'{node.name}.{item.name}')

                # Check top-level functions if requested
                elif check_functions and isinstance(node, ast.FunctionDef):
                    if node.col_offset == 0:  # Top-level only
                        # Skip private functions
                        if node.name.startswith('_'):
                            continue
                        doc = ast.get_docstring(node)
                        if not doc:
                            results[relative_path]['functions'].append(node.name)

        except SyntaxError as e:
            errors.append(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            errors.append(f"Error parsing {file_path}: {e}")

    # Print results
    print("\nFiles with missing docstrings:\n")
    files_needing_work = []
    for file_path in sorted(results.keys()):
        data = results[file_path]
        missing = []

        if not data['module']:
            missing.append("  - Module docstring")

        for cls in data['classes']:
            missing.append(f"  - Class: {cls}")

        for func in data['functions']:
            missing.append(f"  - Function: {func}")

        for method in data['methods']:
            missing.append(f"  - Method: {method}")

        if missing:
            files_needing_work.append((file_path, missing))

    # Display errors if any
    if errors:
        print("\nERRORS ENCOUNTERED:")
        for error in errors:
            print(f"  WARNING: {error}")

    # Display missing docstrings
    if files_needing_work:
        for file_path, missing in files_needing_work:
            print(f"{file_path}")
            for item in missing:
                print(item)
        print(f"\n[FAIL] Total files needing work: {len(files_needing_work)}")
        return False, results
    else:
        print("[PASS] All checked files have necessary docstrings!")
        return True, results


def main():
    """Run comprehensive docstring checks and provide summary."""
    print("\n" + "=" * 70)
    print("PAPYLIO DOCSTRING CHECKER".center(70))
    print("=" * 70 + "\n")

    # Check all papylio files with minimal reporting
    passed, all_results = check_docstrings(
        target_paths=['papylio'],
        check_methods=True,
        check_functions=True,
        description="Checking all papylio files"
    )

    # Summary - show status for each file
    print("\n" + "=" * 70)
    print("FILE STATUS SUMMARY".center(70))
    print("=" * 70)

    for file_path in sorted(all_results.keys()):
        data = all_results[file_path]
        has_issues = (not data['module'] or data['classes'] or data['functions'] or data['methods'])
        status = "FAIL" if has_issues else "PASS"
        print(f"{file_path:<50} {status:>15}")

    print("=" * 70)
    if passed:
        print("[PASS] ALL FILES CHECKED - Project documentation is complete!")
    else:
        print("[FAIL] SOME FILES HAVE MISSING DOCSTRINGS - See details above")
    print("=" * 70 + "\n")

    return passed

def test_docstrings():
    success = main()
    assert success
