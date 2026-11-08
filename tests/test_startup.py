import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import APIRouter


class StartupCacheTests(unittest.TestCase):
    def run_startup(self, path, environment):
        factory = Mock()
        with patch.dict("os.environ", environment, clear=True), patch.dict("sys.modules", {
            "sbv2apilite.tts": SimpleNamespace(StyleBertVits2TTS=factory),
            "sbv2apilite.api": SimpleNamespace(get_api_router=Mock(return_value=APIRouter())),
            "style_bert_vits2.nlp.japanese.user_dict": SimpleNamespace(update_dict=Mock()),
        }), patch("logging.basicConfig"):
            runpy.run_path(str(Path(__file__).resolve().parents[1] / path))
        return factory.call_args.kwargs

    def test_startup_defaults_to_memory(self):
        for path, directory in (("run.py", ".cache/tts"), ("docker/app/run.py", "/cache")):
            with self.subTest(entrypoint=path):
                arguments = self.run_startup(path, {})
                self.assertEqual(arguments["cache_backend"], "memory")
                self.assertEqual(arguments["cache_dir"], directory)

    def test_startup_accepts_file_backend_and_directory(self):
        environment = {
            "SBV2_API_LITE_CACHE_BACKEND": "file",
            "SBV2_API_LITE_CACHE_DIR": "/tmp/sbv2-audio-cache",
        }
        for path in ("run.py", "docker/app/run.py"):
            with self.subTest(entrypoint=path):
                arguments = self.run_startup(path, environment)
                self.assertEqual(arguments["cache_backend"], "file")
                self.assertEqual(arguments["cache_dir"], "/tmp/sbv2-audio-cache")


if __name__ == "__main__":
    unittest.main()
