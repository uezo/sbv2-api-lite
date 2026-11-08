import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from sbv2apilite.cache import FileCache


# Exercise real HTTP routing and cache operations without loading models.
with patch.dict("sys.modules", {
    "style_bert_vits2.nlp": SimpleNamespace(bert_models=Mock()),
    "style_bert_vits2.tts_model": SimpleNamespace(TTSModel=Mock()),
}):
    from sbv2apilite.api import get_api_router
    from sbv2apilite.tts import StyleBertVits2TTS


class CacheAPITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def create_tts(self, backend="memory", model="current"):
        tts = StyleBertVits2TTS.__new__(StyleBertVits2TTS)
        tts.cache = {} if backend == "memory" else FileCache(self.root / model)
        tts.verbose = False
        tts.tts_model = SimpleNamespace(infer=Mock(return_value=(44100, b"\x00\x00" * 32)))
        return tts

    def create_app(self, tts):
        app = FastAPI()
        app.include_router(get_api_router(tts))
        return app

    async def request(self, app, method, target, raise_server_exceptions=True):
        path, _, query = target.partition("?")
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": query.encode(),
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }
        try:
            await app(scope, receive, send)
        except Exception:
            if raise_server_exceptions:
                raise
        status = next(message["status"] for message in messages if message["type"] == "http.response.start")
        body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
        return status, body

    async def test_clear_both_cache_stages_and_regenerate_audio(self):
        for backend in ("memory", "file"):
            with self.subTest(backend=backend):
                tts = self.create_tts(backend)
                app = self.create_app(tts)
                status, _ = await self.request(app, "GET", "/voice?text=hello")
                self.assertEqual(status, 200)
                self.assertEqual(len(tts.cache), 1)
                tts.cache["converted_mp3.mp3"] = b"mp3 audio"
                tts.cache["converted_wav.wav"] = b"converted wav audio"

                status, body = await self.request(app, "DELETE", "/cache")

                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body), {"result": "success"})
                self.assertEqual(len(tts.cache), 0)
                if backend == "file":
                    self.assertEqual(list(tts.cache.directory.iterdir()), [])
                status, _ = await self.request(app, "GET", "/voice?text=hello")
                self.assertEqual(status, 200)
                self.assertEqual(tts.tts_model.infer.call_count, 2)

    async def test_empty_cache_can_be_cleared_repeatedly_without_a_loaded_model(self):
        for backend in ("memory", "file"):
            with self.subTest(backend=backend):
                tts = self.create_tts(backend)
                tts.tts_model = None
                app = self.create_app(tts)
                for _ in range(2):
                    status, body = await self.request(app, "DELETE", "/cache")
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body), {"result": "success"})

    async def test_file_clear_preserves_other_models(self):
        tts = self.create_tts("file")
        other = self.create_tts("file", model="other")
        tts.cache["source_request.wav"] = b"current model audio"
        other.cache["source_request.wav"] = b"other model audio"

        status, _ = await self.request(self.create_app(tts), "DELETE", "/cache")

        self.assertEqual(status, 200)
        self.assertEqual(len(tts.cache), 0)
        self.assertEqual(other.cache["source_request.wav"], b"other model audio")

    async def test_prefixed_router_clears_only_its_own_cache(self):
        first, second = self.create_tts(), self.create_tts()
        first.cache["source_request.wav"] = b"first model audio"
        second.cache["source_request.wav"] = b"second model audio"
        app = self.create_app(first)
        app.include_router(get_api_router(second), prefix="/second")

        status, body = await self.request(app, "DELETE", "/second/cache")

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"result": "success"})
        self.assertEqual(len(second.cache), 0)
        self.assertEqual(first.cache["source_request.wav"], b"first model audio")

    async def test_clear_failure_returns_an_error(self):
        tts = self.create_tts()
        tts.cache = Mock()
        tts.cache.clear.side_effect = PermissionError("cache directory is not writable")

        status, _ = await self.request(
            self.create_app(tts), "DELETE", "/cache", raise_server_exceptions=False
        )

        self.assertEqual(status, 500)


if __name__ == "__main__":
    unittest.main()
