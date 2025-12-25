import os
import pytest

@pytest.hookimpl
def pytest_collection_modifyitems(config, items):
    """
    Filter out LAB tests unless CATDPT_LAB environment variable is set.
    """
    if os.environ.get("CATDPT_LAB") == "1":
        return

    # Filter in-place
    new_items = []
    for item in items:
        # Check if 'LAB' is in the file path
        path_str = str(item.path).replace('\\', '/')
        if '/LAB/' in f'/{path_str}/':
            continue
        new_items.append(item)
    
    items[:] = new_items
