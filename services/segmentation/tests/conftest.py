"""
Pytest configuration for Segmentation service tests.
"""

import sys
import os

# Make `app` (service root) and `shared` (services/shared) importable
SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICES_ROOT = os.path.dirname(SERVICE_ROOT)
for path in (SERVICES_ROOT, SERVICE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
