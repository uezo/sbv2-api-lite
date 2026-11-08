import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from style_bert_vits2.constants import Languages


with patch.dict("sys.modules", {
    "style_bert_vits2.nlp": SimpleNamespace(bert_models=Mock()),
    "style_bert_vits2.tts_model": SimpleNamespace(TTSModel=Mock()),
}):
    from sbv2apilite import tts as tts_module


class LanguageModelLoadingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "model.safetensors").write_bytes(b"voice model")
        (self.model_dir / "style_vectors.npy").write_bytes(b"style vectors")
        self.write_config("2.7.0-JP-Extra")
        self.bert = SimpleNamespace(load_model=Mock(), load_tokenizer=Mock())
        self.factory = Mock(side_effect=self.make_model)
        for name, value in (("bert_models", self.bert), ("TTSModel", self.factory)):
            patcher = patch.object(tts_module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_config(self, version):
        (self.model_dir / "config.json").write_text(json.dumps({"version": version}))

    def make_model(self, **kwargs):
        version = json.loads(Path(kwargs["config_path"]).read_text())["version"]
        return SimpleNamespace(hyper_parameters=SimpleNamespace(version=version))

    def create_tts(self):
        return tts_module.StyleBertVits2TTS(model_dir=str(self.model_dir))

    def assert_normal_model_resources_prepared(self):
        expected = [
            call(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm"),
            call(Languages.EN, "microsoft/deberta-v3-large"),
            call(Languages.ZH, "hfl/chinese-roberta-wwm-ext-large"),
        ]
        self.assertEqual(self.bert.load_model.call_args_list, expected)
        self.assertEqual(self.bert.load_tokenizer.call_args_list, expected)

    def test_jp_extra_prepares_only_japanese_resources(self):
        tts = self.create_tts()

        self.assertIsNotNone(tts.tts_model)
        self.bert.load_model.assert_called_once_with(
            Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm"
        )
        self.bert.load_tokenizer.assert_called_once_with(
            Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm"
        )

    def test_normal_model_prepares_all_languages_with_explicit_repository_ids(self):
        self.write_config("2.0")

        tts = self.create_tts()

        self.assertIsNotNone(tts.tts_model)
        self.assert_normal_model_resources_prepared()

    def test_switching_to_normal_model_prepares_english_and_chinese(self):
        tts = self.create_tts()
        self.bert.load_model.reset_mock()
        self.bert.load_tokenizer.reset_mock()
        self.write_config("2.0")

        self.assertTrue(tts.load_tts_model())

        self.assert_normal_model_resources_prepared()

    def test_resource_failure_fails_model_load_and_can_be_retried(self):
        self.write_config("2.0")
        for method in ("load_model", "load_tokenizer"):
            for failing_language in (Languages.EN, Languages.ZH):
                with self.subTest(method=method, language=failing_language):
                    self.bert.load_model.reset_mock(side_effect=True)
                    self.bert.load_tokenizer.reset_mock(side_effect=True)

                    def prepare(language, model_name):
                        if language == failing_language:
                            raise OSError("resource download failed")

                    getattr(self.bert, method).side_effect = prepare
                    with self.assertLogs(tts_module.logger, level="ERROR"):
                        tts = self.create_tts()

                    self.assertIsNone(tts.tts_model)
                    self.assertEqual(len(tts.cache), 0)
                    self.bert.load_model.reset_mock(side_effect=True)
                    self.bert.load_tokenizer.reset_mock(side_effect=True)

                    self.assertTrue(tts.load_tts_model())
                    self.assertIsNotNone(tts.tts_model)
                    self.assert_normal_model_resources_prepared()

    def test_missing_voice_model_does_not_prepare_language_resources(self):
        (self.model_dir / "model.safetensors").unlink()

        tts = self.create_tts()

        self.assertIsNone(tts.tts_model)
        self.factory.assert_not_called()
        self.bert.load_model.assert_not_called()
        self.bert.load_tokenizer.assert_not_called()

    def test_import_does_not_prepare_language_resources(self):
        with patch.dict("sys.modules", {
            "style_bert_vits2.nlp": SimpleNamespace(bert_models=self.bert),
            "style_bert_vits2.tts_model": SimpleNamespace(TTSModel=self.factory),
        }):
            spec = importlib.util.spec_from_file_location(
                "sbv2apilite._tts_import_test", tts_module.__file__
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        self.bert.load_model.assert_not_called()
        self.bert.load_tokenizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
