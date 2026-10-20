#!/usr/bin/env python3
"""
AgriEdge Intelligence OS — Root Application Launcher
══════════════════════════════════════════════════════
Run this from the project root:
    python3 run.py
or
    python3 app.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app
from backend.config import HOST, PORT, DEBUG

def start_server():
    target_port = int(os.environ.get("PORT", PORT))
    print("\n" + "=" * 60)
    print("  🌱 AGRI-EDGE INTELLIGENCE OPERATING SYSTEM")
    print("=" * 60)
    try:
        print(f"  • Host: http://{HOST}:{target_port}")
        print(f"  • Open in your browser: http://127.0.0.1:{target_port}")
        print("  • Press CTRL+C to stop\n")
        app.run(host=HOST, port=target_port, debug=DEBUG)
    except OSError as e:
        if "Address already in use" in str(e) or getattr(e, 'errno', 0) == 48:
            fallback_port = 5050
            print(f"\n[!] Port {target_port} is occupied (common on macOS AirPlay Receiver).")
            print(f"    Switching automatically to fallback port: http://{HOST}:{fallback_port}")
            print(f"    Open in your browser: http://127.0.0.1:{fallback_port}\n")
            app.run(host=HOST, port=fallback_port, debug=DEBUG)
        else:
            raise e

if __name__ == '__main__':
    start_server()
