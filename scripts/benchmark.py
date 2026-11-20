import sys
from pathlib import Path

# Allow direct execution from an uninstalled checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fluxhermes.cli.benchmark import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
