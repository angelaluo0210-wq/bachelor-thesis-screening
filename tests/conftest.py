"""Never allow the repository's sealed test split to be opened by pytest."""
import os
import sys
from pathlib import Path

_SEALED_TEST = (Path(__file__).resolve().parents[1] / "data/processed/splits/test.csv").resolve()


def _protect_sealed_test(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        if Path(os.fsdecode(args[0])).resolve() == _SEALED_TEST:
            raise PermissionError("The real test.csv is sealed during all tests")


sys.addaudithook(_protect_sealed_test)
