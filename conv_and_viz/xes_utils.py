import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


GZIP_MAGIC = b"\x1f\x8b"


def is_gzipped_file(path: str) -> bool:
    """Detect gzip-compressed content from the file header."""
    file_path = Path(path)
    if not file_path.is_file():
        return False
    with file_path.open("rb") as handle:
        return handle.read(2) == GZIP_MAGIC


@contextmanager
def normalized_xes_path(xes_path: str) -> Iterator[str]:
    """
    Yield a PM4Py-friendly path for XES imports.

    Some public logs are gzip-compressed but still distributed with a `.xes`
    filename. PM4Py relies on the `.gz` suffix to auto-detect compressed XES,
    so create a temporary sibling alias when needed.
    """
    source = Path(xes_path)
    alias_path = None
    try:
        if is_gzipped_file(str(source)) and source.suffix.lower() != ".gz":
            alias_path = source.with_name(f"{source.name}.gz")
            if alias_path.exists():
                alias_path.unlink()
            shutil.copyfile(source, alias_path)
            yield str(alias_path)
        else:
            yield str(source)
    finally:
        if alias_path is not None and alias_path.exists():
            alias_path.unlink()
