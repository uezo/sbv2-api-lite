import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from style_bert_vits2.constants import Languages


# Import the real cache implementation without loading or downloading models.
with patch.dict("sys.modules", {
    "style_bert_vits2.nlp": SimpleNamespace(bert_models=Mock()),
    "style_bert_vits2.tts_model": SimpleNamespace(TTSModel=Mock()),
}):
    from sbv2apilite.tts import MP3ConvertProcessor, StyleBertVits2TTS


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

        wave = await self.tts.tts(**self.parameters)
        mp3 = await self.tts.tts(**self.parameters, post_processor=processor)
        cached_mp3 = await self.tts.tts(**self.parameters, post_processor=processor)

        self.assertTrue(wave.startswith(b"RIFF"))
        self.assertEqual(mp3, b"mp3 audio")
        self.assertEqual(cached_mp3, mp3)
        self.assertEqual(self.tts.tts_model.infer.call_count, 2)
        processor.process.assert_awaited_once()

    def test_underscores_do_not_cause_key_collisions(self):
        first = self.tts.generate_cache_key("hello_0", 1, "Neutral")
        second = self.tts.generate_cache_key("hello", 0, "1_Neutral")

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
