"""Launch the local SAKSHI presentation app or the legacy Streamlit view."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    from sakshi.app.web_server import main as serve

    parser = argparse.ArgumentParser(description="SAKSHI local fieldroom")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    serve(port=args.port)


def streamlit_main() -> None:
    app = Path(__file__).with_name("app") / "streamlit_app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app), "--server.address", "127.0.0.1", *sys.argv[1:]]
    try:
        returncode = subprocess.call(command)
    except KeyboardInterrupt:
        returncode = 130
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
