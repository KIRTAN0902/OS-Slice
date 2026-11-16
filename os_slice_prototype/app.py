"""
OS-Slice Interactive Research Lab & Evaluation Console
Delegates to the redesigned Research Lab in research_lab/app.py so existing
launch scripts (run_ui.ps1, oslice_cli.py) work seamlessly.
"""

import sys
from pathlib import Path

_PROTO_ROOT = Path(__file__).resolve().parent
if str(_PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTO_ROOT))

from research_lab.app import main

if __name__ == "__main__":
    main()
