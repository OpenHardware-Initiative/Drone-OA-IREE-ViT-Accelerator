# HOW TO import packages from a folder?

## `.ipynb` files

We dont need to change the export as long as we opened the folder in vscode.

Only change we need to the third party repo is to add a `.` to the import statement:

```python
from .ViTSubmodules import *
```

## `.py` files

In `.py` files, we can use the same approach as in the notebook, but we need to ensure that the project root is correctly identified. Here's how you can do it:

```python
import sys
import os

# --- Start of the fix ---
# Programmatically find the project root and add it to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of the fix ---

# Now your regular imports will work
from third_party.vitfly.models.ViTsubmodules import *
from third_party.vitfly.models.model import *
```