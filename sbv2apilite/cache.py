from collections.abc import MutableMapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator


class FileCache(MutableMapping[str, bytes]):
    """A dictionary-compatible audio cache scoped to one model directory."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not key or key in (".", "..") or Path(key).name != key or key.endswith(".tmp"):
            raise ValueError("Cache keys must be filenames without a .tmp suffix")
        return self.directory / key

    def __getitem__(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: bytes):
        target = self._path(key)
        temporary_path = None
        try:
            # Readers only see complete files, including across worker processes.
            with NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as f:
                temporary_path = Path(f.name)
                f.write(value)
            temporary_path.replace(target)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def __delitem__(self, key: str):
        try:
            self._path(key).unlink()
        except FileNotFoundError:
            raise KeyError(key) from None

    def __iter__(self) -> Iterator[str]:
        return (
            path.name for path in self.directory.iterdir()
            if path.is_file() and path.suffix != ".tmp"
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def clear(self):
        # Leave other model directories and in-progress writes alone.
        for key in self:
            self._path(key).unlink(missing_ok=True)
