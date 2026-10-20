#!/usr/bin/env python3
"""
AgriEdge Intelligence OS — Root Application Entrypoint
══════════════════════════════════════════════════════
Run this from the project root:
    python3 app.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run import start_server

if __name__ == '__main__':
    start_server()
