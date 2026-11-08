import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sbv2apilite.cache import FileCache


class FileCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = FileCache(self.root / "model-a")

    def test_dictionary_operations_and_persistence(self):
        self.assertIsNone(self.cache.get("missing"))
        with self.assertRaises(KeyError):
            _ = self.cache["missing"]

        self.cache["request"] = b"audio"
        reopened = FileCache(self.cache.directory)
        self.assertEqual(reopened["request"], b"audio")
        self.assertEqual(list(reopened), ["request"])
        self.assertEqual(len(reopened), 1)

        del reopened["request"]
        self.assertNotIn("request", self.cache)
        with self.assertRaises(KeyError):
            del reopened["request"]

    def test_clear_is_scoped_to_one_model(self):
        other = FileCache(self.root / "model-b")
        self.cache["request"] = b"model a"
        other["request"] = b"model b"
        temporary_file = self.cache.directory / "pending.tmp"
        temporary_file.write_bytes(b"pending")

        self.cache.clear()
        self.cache.clear()

        self.assertEqual(len(self.cache), 0)
        self.assertEqual(other["request"], b"model b")
        self.assertEqual(temporary_file.read_bytes(), b"pending")

    def test_readers_see_complete_files_during_replacement(self):
        self.cache["request"] = b"old audio"
        replace = Path.replace

        def check_and_replace(temporary_path, target):
            self.assertEqual(temporary_path.read_bytes(), b"new audio")
            self.assertEqual(self.cache["request"], b"old audio")
            return replace(temporary_path, target)

        with patch.object(Path, "replace", check_and_replace):
            self.cache["request"] = b"new audio"

        self.assertEqual(self.cache["request"], b"new audio")
        self.assertEqual(list(self.cache.directory.glob("*.tmp")), [])

    def test_failed_write_preserves_previous_audio_and_removes_temporary_file(self):
        self.cache["request"] = b"old audio"
        with patch.object(Path, "replace", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                self.cache["request"] = b"new audio"

        self.assertEqual(self.cache["request"], b"old audio")
        self.assertEqual(list(self.cache.directory.glob("*.tmp")), [])

    def test_keys_are_stored_as_filenames_without_an_added_extension(self):
        self.cache["source_request.wav"] = b"wav audio"
        self.cache["converted_request.mp3"] = b"mp3 audio"
        self.cache["converted_other.wav"] = b"converted wav audio"

        self.assertEqual(
            {path.name for path in self.cache.directory.iterdir()},
            {"source_request.wav", "converted_request.mp3", "converted_other.wav"},
        )
        self.assertEqual(self.cache["converted_other.wav"], b"converted wav audio")

    def test_keys_cannot_escape_the_model_directory(self):
        for key in ("../other.wav", "/tmp/other.wav", "", ".", "..", "reserved.tmp"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.cache[key] = b"audio"


if __name__ == "__main__":
    unittest.main()
