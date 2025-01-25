import sys
import os

# Add the project directory to sys.path
project_path = os.path.abspath('/path/to/your/project')
if project_path not in sys.path:
    sys.path.append(project_path)
