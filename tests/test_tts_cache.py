import asyncio
import io
import shutil
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from style_bert_vits2.constants import Languages


# Import the real cache implementation without loading or downloading models.
with patch.dict("sys.modules", {
    "style_bert_vits2.nlp": SimpleNamespace(bert_models=Mock()),
    "style_bert_vits2.tts_model": SimpleNamespace(TTSModel=Mock()),
}):
    from sbv2apilite import tts as tts_module

MP3ConvertProcessor = tts_module.MP3ConvertProcessor
PostProcessor = tts_module.PostProcessor
StyleBertVits2TTS = tts_module.StyleBertVits2TTS
FileCache = tts_module.FileCache


class TTSCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tts = StyleBertVits2TTS.__new__(StyleBertVits2TTS)
        self.tts.cache = {}
        self.tts.verbose = False
        self.tts.tts_model = SimpleNamespace(
            infer=Mock(return_value=(44100, b"\x00\x00" * 32))
        )
        self.parameters = {
            "text": "こんにちは",
            "speaker_id": 0,
            "style": "Neutral",
            "language": Languages.JP,
            "sdp_ratio": 0.2,
            "noise": 0.6,
            "noise_w": 0.8,
            "length": 1.0,
            "line_split": True,
            "split_interval": 0.5,
            "assist_text": None,
            "assist_text_weight": 1.0,
            "style_weight": 1.0,
            "reference_audio_path": None,
        }

    async def test_identical_requests_reuse_cached_audio(self):
        first = await self.tts.tts(**self.parameters)
        second = await self.tts.tts(**self.parameters)

        self.assertEqual(first, second)
        self.tts.tts_model.infer.assert_called_once_with(**self.parameters)
        self.assertEqual(len(self.tts.cache), 1)
        self.assertTrue(all(key.startswith("source_") and key.endswith(".wav") for key in self.tts.cache))

    async def test_parameter_changes_generate_new_audio(self):
        changes = {
            "text": "こんばんは",
            "speaker_id": 1,
            "style": "Happy",
            "language": Languages.EN,
            "sdp_ratio": 0.3,
            "noise": 0.7,
            "noise_w": 0.9,
            "length": 1.5,
            "line_split": False,
            "split_interval": 0.8,
            "assist_text": "楽しそうに",
            "assist_text_weight": 0.5,
            "style_weight": 2.0,
            "reference_audio_path": "reference.wav",
            "pitch_scale": 1.2,
            "given_phone": ["k", "o"],
        }
        for name, value in changes.items():
            with self.subTest(parameter=name):
                self.tts.clear_cache()
                self.tts.tts_model.infer.reset_mock()

                await self.tts.tts(**self.parameters)
                changed = {**self.parameters, name: value}
                await self.tts.tts(**changed)
                await self.tts.tts(**changed)

                self.assertEqual(self.tts.tts_model.infer.call_count, 2)
                self.tts.tts_model.infer.assert_called_with(**changed)
                self.assertEqual(len(self.tts.cache), 2)

    async def test_parameter_order_does_not_change_cache_key(self):
        await self.tts.tts(**self.parameters)
        await self.tts.tts(**dict(reversed(self.parameters.items())))

        self.tts.tts_model.infer.assert_called_once()

    async def test_wave_and_mp3_have_separate_cache_entries(self):
        processor = MP3ConvertProcessor()
        processor.process = AsyncMock(return_value=b"mp3 audio")

        source_audio = await self.tts.tts(**self.parameters)
        mp3 = await self.tts.tts(**self.parameters, post_processor=processor)
        cached_mp3 = await self.tts.tts(**self.parameters, post_processor=processor)

        self.assertTrue(source_audio.startswith(b"RIFF"))
        self.assertEqual(mp3, b"mp3 audio")
        self.assertEqual(cached_mp3, mp3)
        self.tts.tts_model.infer.assert_called_once()
        processor.process.assert_awaited_once_with(source_audio, 44100, False)
        self.assertEqual(len(self.tts.cache), 2)

    async def test_mp3_first_also_caches_the_source_wav(self):
        processor = MP3ConvertProcessor()
        processor.process = AsyncMock(return_value=b"mp3 audio")

        await self.tts.tts(**self.parameters, post_processor=processor)
        source_audio = await self.tts.tts(**self.parameters)

        self.assertTrue(source_audio.startswith(b"RIFF"))
        self.tts.tts_model.infer.assert_called_once()
        processor.process.assert_awaited_once_with(source_audio, 44100, False)
        self.assertEqual(len(self.tts.cache), 2)

    async def test_wav_to_wav_preserves_both_versions(self):
        class WavProcessor(PostProcessor):
            file_extension = "wav"

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)
            wf.setframerate(22050)
            wf.writeframes(b"\x80" * 32)
        converted_wav = buffer.getvalue()
        processor = WavProcessor()
        processor.process = AsyncMock(return_value=converted_wav)

        converted_audio = await self.tts.tts(**self.parameters, post_processor=processor)
        source_audio = await self.tts.tts(**self.parameters)
        cached_conversion = await self.tts.tts(**self.parameters, post_processor=processor)

        self.assertEqual(converted_audio, converted_wav)
        self.assertEqual(cached_conversion, converted_wav)
        self.assertNotEqual(source_audio, converted_audio)
        self.tts.tts_model.infer.assert_called_once()
        processor.process.assert_awaited_once_with(source_audio, 44100, False)
        self.assertEqual(len(self.tts.cache), 2)
        for key, data in self.tts.cache.items():
            self.assertTrue(key.endswith(".wav"))
            with wave.open(io.BytesIO(data), "rb") as wf:
                if key.startswith("source_"):
                    self.assertEqual((wf.getframerate(), wf.getsampwidth()), (44100, 2))
                else:
                    self.assertTrue(key.startswith("converted_"))
                    self.assertEqual((wf.getframerate(), wf.getsampwidth()), (22050, 1))

    async def test_custom_output_extension_is_used(self):
        class OggProcessor(PostProcessor):
            file_extension = "ogg"

        processor = OggProcessor()
        processor.process = AsyncMock(return_value=b"ogg audio")

        audio = await self.tts.tts(**self.parameters, post_processor=processor)

        self.assertEqual(audio, b"ogg audio")
        self.assertEqual(sum(key.startswith("source_") for key in self.tts.cache), 1)
        self.assertEqual(sum(key.startswith("converted_") and key.endswith(".ogg") for key in self.tts.cache), 1)

    async def test_converted_cache_hit_does_not_require_source_wav(self):
        processor = MP3ConvertProcessor()
        processor.process = AsyncMock(return_value=b"mp3 audio")
        await self.tts.tts(**self.parameters, post_processor=processor)
        source_key = next(key for key in self.tts.cache if key.startswith("source_"))
        del self.tts.cache[source_key]
        self.tts.tts_model.infer.reset_mock()
        processor.process.reset_mock()

        audio = await self.tts.tts(**self.parameters, post_processor=processor)

        self.assertEqual(audio, b"mp3 audio")
        self.tts.tts_model.infer.assert_not_called()
        processor.process.assert_not_awaited()
        self.assertEqual(len(self.tts.cache), 1)

    async def test_conversion_failure_keeps_source_for_retry(self):
        processor = MP3ConvertProcessor()
        processor.process = AsyncMock(side_effect=[RuntimeError("conversion failed"), b"mp3 audio"])

        with self.assertLogs(tts_module.logger, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                await self.tts.tts(**self.parameters, post_processor=processor)
        self.assertEqual(len(self.tts.cache), 1)
        self.assertTrue(all(key.startswith("source_") for key in self.tts.cache))

        audio = await self.tts.tts(**self.parameters, post_processor=processor)

        self.assertEqual(audio, b"mp3 audio")
        self.tts.tts_model.infer.assert_called_once()
        self.assertEqual(processor.process.await_count, 2)
        self.assertEqual(len(self.tts.cache), 2)

    async def test_synthesis_changes_invalidate_both_stages(self):
        processor = MP3ConvertProcessor()
        processor.process = AsyncMock(return_value=b"mp3 audio")
        await self.tts.tts(**self.parameters, post_processor=processor)

        for name, value in (("language", Languages.EN), ("length", 1.5)):
            changed = {**self.parameters, name: value}
            await self.tts.tts(**changed, post_processor=processor)
            await self.tts.tts(**changed, post_processor=processor)

        self.assertEqual(self.tts.tts_model.infer.call_count, 3)
        self.assertEqual(processor.process.await_count, 3)
        self.assertEqual(len(self.tts.cache), 6)

    async def test_clear_removes_source_and_converted_audio(self):
        processor = MP3ConvertProcessor()
        processor.process = AsyncMock(return_value=b"mp3 audio")
        await self.tts.tts(**self.parameters, post_processor=processor)

        self.tts.clear_cache()

        self.assertEqual(len(self.tts.cache), 0)
        await self.tts.tts(**self.parameters, post_processor=processor)
        self.assertEqual(self.tts.tts_model.infer.call_count, 2)
        self.assertEqual(processor.process.await_count, 2)

    def test_underscores_do_not_cause_key_collisions(self):
        first = self.tts.generate_cache_key("hello_0", 1, "Neutral")
        second = self.tts.generate_cache_key("hello", 0, "1_Neutral")

        self.assertNotEqual(first, second)

    async def test_mp3_configuration_changes_generate_separate_audio(self):
        for bitrate, ffmpeg_path in (("64k", "ffmpeg"), ("128k", "ffmpeg"), ("64k", "/other/ffmpeg")):
            with self.subTest(bitrate=bitrate, ffmpeg_path=ffmpeg_path):
                processor = MP3ConvertProcessor(bitrate, ffmpeg_path)
                expected = f"{bitrate}:{ffmpeg_path}".encode()
                processor.process = AsyncMock(return_value=expected)

                first = await self.tts.tts(**self.parameters, post_processor=processor)
                second = await self.tts.tts(**self.parameters, post_processor=processor)

                self.assertEqual(first, expected)
                self.assertEqual(second, expected)
                processor.process.assert_awaited_once()

        self.tts.tts_model.infer.assert_called_once()
        self.assertEqual(len(self.tts.cache), 4)


class FileTTSCacheTests(TTSCacheTests):
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tts.cache = FileCache(Path(temporary.name) / "model")

    async def test_files_have_playable_audio_extensions(self):
        processor = MP3ConvertProcessor()
        processor.process = AsyncMock(return_value=b"mp3 audio")

        await self.tts.tts(**self.parameters, post_processor=processor)

        files = list(self.tts.cache.directory.iterdir())
        self.assertEqual(len(files), 2)
        source = next(path for path in files if path.name.startswith("source_"))
        converted = next(path for path in files if path.name.startswith("converted_"))
        self.assertEqual(source.suffix, ".wav")
        self.assertEqual(converted.suffix, ".mp3")
        with wave.open(str(source), "rb") as wf:
            self.assertEqual(wf.getframerate(), 44100)
            self.assertEqual(wf.readframes(wf.getnframes()), b"\x00\x00" * 32)
        self.assertEqual(converted.read_bytes(), b"mp3 audio")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    async def test_saved_mp3_can_be_decoded_and_reused(self):
        self.tts.tts_model.infer.return_value = (44100, b"\x00\x00" * 4410)
        processor = MP3ConvertProcessor()
        processor.process = AsyncMock(wraps=processor.process)

        first = await self.tts.tts(**self.parameters, post_processor=processor)
        second = await self.tts.tts(**self.parameters, post_processor=processor)
        source = await self.tts.tts(**self.parameters)

        self.assertEqual(first, second)
        self.assertTrue(source.startswith(b"RIFF"))
        self.tts.tts_model.infer.assert_called_once()
        processor.process.assert_awaited_once_with(source, 44100, False)
        converted = next(self.tts.cache.directory.glob("converted_*.mp3"))
        decoder = await asyncio.create_subprocess_exec(
            shutil.which("ffmpeg"), "-v", "error", "-i", str(converted), "-f", "null", "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await decoder.communicate()
        self.assertEqual(decoder.returncode, 0, stderr.decode())


class ModelCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        for name in ("model.safetensors", "config.json", "style_vectors.npy"):
            (self.model_dir / name).write_bytes(name.encode())
        self.cache_dir = self.root / "cache"
        self.models = []
        factory = patch.object(tts_module, "TTSModel", side_effect=self.make_model)
        factory.start()
        self.addCleanup(factory.stop)

    def make_model(self, **kwargs):
        model = SimpleNamespace(
            infer=Mock(return_value=(44100, bytes([len(self.models) + 1, 0]) * 32)),
            hyper_parameters=SimpleNamespace(version="2.7.0-JP-Extra"),
        )
        self.models.append(model)
        return model

    def create_tts(self, **kwargs):
        return StyleBertVits2TTS(
            model_dir=str(self.model_dir),
            cache_dir=str(self.cache_dir),
            **kwargs,
        )

    async def test_default_memory_cache_creates_no_files_and_resets_on_load(self):
        tts = self.create_tts()
        self.assertIsInstance(tts.cache, dict)
        self.assertFalse(self.cache_dir.exists())
        first = await tts.tts("こんにちは", 0, "Neutral")
        self.assertEqual(len(tts.cache), 1)

        self.assertTrue(tts.load_tts_model())
        self.assertEqual(len(tts.cache), 0)
        second = await tts.tts("こんにちは", 0, "Neutral")

        self.assertNotEqual(first, second)
        self.assertFalse(self.cache_dir.exists())

    def test_invalid_backend_is_rejected_at_startup(self):
        with self.assertRaisesRegex(ValueError, "cache_backend"):
            self.create_tts(cache_backend="invalid")
        self.assertFalse(self.cache_dir.exists())

    async def test_file_cache_survives_restart_and_model_reload(self):
        first = self.create_tts(cache_backend="file")
        audio = await first.tts("こんにちは", 0, "Neutral")
        directory = first.cache.directory

        second = self.create_tts(cache_backend="file")
        self.assertEqual(second.cache.directory, directory)
        self.assertEqual(await second.tts("こんにちは", 0, "Neutral"), audio)
        second.tts_model.infer.assert_not_called()

        self.assertTrue(second.load_tts_model())
        self.assertEqual(await second.tts("こんにちは", 0, "Neutral"), audio)
        second.tts_model.infer.assert_not_called()

    async def test_converted_audio_and_source_are_reusable_after_restart(self):
        first = self.create_tts(cache_backend="file")
        processor = MP3ConvertProcessor()
        processor.process = AsyncMock(return_value=b"mp3 audio")
        await first.tts("こんにちは", 0, "Neutral", post_processor=processor)

        second = self.create_tts(cache_backend="file")
        processor = MP3ConvertProcessor()
        processor.process = AsyncMock(return_value=b"should use cached conversion")
        self.assertEqual(
            await second.tts("こんにちは", 0, "Neutral", post_processor=processor),
            b"mp3 audio",
        )
        processor.process.assert_not_awaited()

        new_processor = MP3ConvertProcessor(bitrate="128k")
        new_processor.process = AsyncMock(return_value=b"new bitrate audio")
        self.assertEqual(
            await second.tts("こんにちは", 0, "Neutral", post_processor=new_processor),
            b"new bitrate audio",
        )
        source = await second.tts("こんにちは", 0, "Neutral")
        new_processor.process.assert_awaited_once_with(source, 44100, False)
        second.tts_model.infer.assert_not_called()
        self.assertEqual(len(second.cache), 3)

        self.assertTrue(second.load_tts_model())
        self.assertEqual(await second.tts("こんにちは", 0, "Neutral"), source)
        second.tts_model.infer.assert_not_called()

    async def test_changing_any_model_file_selects_a_separate_cache(self):
        for name in ("model.safetensors", "config.json", "style_vectors.npy"):
            with self.subTest(file=name):
                tts = self.create_tts(cache_backend="file")
                original_audio = await tts.tts("こんにちは", 0, "Neutral")
                original_cache = tts.cache
                path = self.model_dir / name
                original_contents = path.read_bytes()
                path.write_bytes(original_contents + b"changed")

                self.assertTrue(tts.load_tts_model())
                self.assertNotEqual(tts.cache.directory, original_cache.directory)
                changed_audio = await tts.tts("こんにちは", 0, "Neutral")
                self.assertNotEqual(changed_audio, original_audio)
                tts.tts_model.infer.assert_called_once()
                self.assertEqual(len(original_cache), 1)

                path.write_bytes(original_contents)
                self.assertTrue(tts.load_tts_model())
                self.assertEqual(tts.cache.directory, original_cache.directory)
                self.assertEqual(await tts.tts("こんにちは", 0, "Neutral"), original_audio)
                tts.tts_model.infer.assert_not_called()

    async def test_clear_leaves_other_models_cached(self):
        first = self.create_tts(cache_backend="file")
        await first.tts("こんにちは", 0, "Neutral")
        (self.model_dir / "model.safetensors").write_bytes(b"different model")
        second = self.create_tts(cache_backend="file")
        second_audio = await second.tts("こんにちは", 0, "Neutral")
        second.tts_model.infer.reset_mock()

        first.clear_cache()

        self.assertEqual(len(first.cache), 0)
        self.assertEqual(await second.tts("こんにちは", 0, "Neutral"), second_audio)
        second.tts_model.infer.assert_not_called()

    async def test_file_cache_can_be_initialized_before_a_model_is_uploaded(self):
        for path in self.model_dir.iterdir():
            path.unlink()
        tts = self.create_tts(cache_backend="file")
        self.assertIsNone(tts.tts_model)

        for name in ("model.safetensors", "config.json", "style_vectors.npy"):
            (self.model_dir / name).write_bytes(name.encode())
        self.assertTrue(tts.load_tts_model())
        await tts.tts("こんにちは", 0, "Neutral")
        self.assertIsInstance(tts.cache, FileCache)
        self.assertEqual(len(tts.cache), 1)

    async def test_in_flight_audio_stays_with_its_original_model_cache(self):
        for backend in ("memory", "file"):
            with self.subTest(backend=backend):
                tts = self.create_tts(cache_backend=backend)
                original_cache = tts.cache
                original_model = tts.tts_model
                original_output = original_model.infer.return_value
                started, release = threading.Event(), threading.Event()

                def infer(**kwargs):
                    started.set()
                    if not release.wait(timeout=5):
                        raise TimeoutError("test did not release inference")
                    return original_output

                original_model.infer.side_effect = infer
                pending = asyncio.create_task(tts.tts("こんにちは", 0, "Neutral"))
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 2))
                    path = self.model_dir / "model.safetensors"
                    path.write_bytes(path.read_bytes() + b"changed")
                    self.assertTrue(tts.load_tts_model())
                    new_audio = await tts.tts("こんにちは", 0, "Neutral")
                finally:
                    release.set()
                    old_audio = await pending

                self.assertNotEqual(old_audio, new_audio)
                self.assertEqual(list(original_cache.values()), [old_audio])
                self.assertEqual(list(tts.cache.values()), [new_audio])
                self.assertEqual(await tts.tts("こんにちは", 0, "Neutral"), new_audio)
                tts.tts_model.infer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
