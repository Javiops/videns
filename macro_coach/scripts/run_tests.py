"""Lanza la suite. Uso: .venv\\Scripts\\python.exe scripts\\run_tests.py [args pytest]"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.exit(subprocess.call([sys.executable, "-m", "pytest",
                          str(ROOT / "tests"), "-q"] + sys.argv[1:],
                         cwd=str(ROOT)))
