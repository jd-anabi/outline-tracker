"""Rules of the repository itself: entry point, import boundaries, reference copies."""

import subprocess
import sys


def _run(code: str) -> str:
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_version_flag():
    # In a subprocess: pytest-qt has already imported PySide6 into this process.
    out = _run(
        "import sys\n"
        "from outline_tracker.cli import main\n"
        "try:\n"
        "    main(['--version'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "heavy = ('torch', 'transformers', 'PySide6', 'pyqtgraph')\n"
        "print('LOADED', sorted(m for m in heavy if m in sys.modules))\n"
    )
    lines = out.splitlines()
    assert lines[0].startswith("outline-tracker 0.1.0"), out
    assert lines[-1] == "LOADED []", out
