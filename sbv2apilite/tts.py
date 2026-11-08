import asyncio
import hashlib
import io
import json
import logging
from pathlib import Path
from threading import Lock
from time import time
import wave
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
from .cache import FileCache


# Get logger
logger = logging.getLogger(__name__)


_BERT_MODEL_NAMES = {
    Languages.JP: "ku-nlp/deberta-v2-large-japanese-char-wwm",
    Languages.EN: "microsoft/deberta-v3-large",
    Languages.ZH: "hfl/chinese-roberta-wwm-ext-large",
}
_BERT_LOAD_LOCK = Lock()


def _prepare_english_nltk_data():
    import nltk

    resources = (
        ("cmudict", "corpora/cmudict.zip"),
        # g2p_en checks for this legacy archive even with current NLTK versions.
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger.zip"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng/"),
    )
    for name, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading English NLTK resource: %s", name)
            if not nltk.download(name, quiet=True, raise_on_error=True):
                raise RuntimeError(f"Failed to download NLTK resource: {name}")
            nltk.data.find(path)


def _prepare_bert_models(version: str):
    # Normal models support all three languages; JP-Extra is Japanese-only.
    languages = (Languages.JP,) if version.endswith("JP-Extra") else tuple(_BERT_MODEL_NAMES)
    # The library caches these resources globally, across TTS instances.
    with _BERT_LOAD_LOCK:
        for language in languages:
            if language == Languages.EN:
                _prepare_english_nltk_data()
            model_name = _BERT_MODEL_NAMES[language]
            logger.info("Preparing %s BERT model and tokenizer from %s", language, model_name)
            # Explicit repository IDs let Hugging Face reuse cached files or
            # download missing assets instead of requiring the default local path.
            bert_models.load_model(language, model_name)
            bert_models.load_tokenizer(language, model_name)
            logger.info("%s BERT model and tokenizer are ready", language)


class PostProcessor:
    file_extension = "bin"

    def cache_key(self) -> str:
        return self.__class__.__name__

    async def process(self, audio_bytes: bytes, sample_rate: int, verbose: bool = False) -> bytes:
        raise NotImplementedError("PostProcessor.process must be implemented")


class MP3ConvertProcessor(PostProcessor):
    file_extension = "mp3"

    def __init__(self, bitrate: str = "64k", ffmpeg_path: str = "ffmpeg"):
        self.bitrate = bitrate
        self.ffmpeg_path = ffmpeg_path

    def cache_key(self) -> str:
        return json.dumps([super().cache_key(), self.bitrate, self.ffmpeg_path])

    async def process(self, audio_bytes: bytes, sample_rate: int, verbose: bool = False) -> bytes:
        start_time = time()
        # Note: This converter doesn't use sample rate because it is included in the wave header
        ffmpeg_proc = await asyncio.create_subprocess_exec(
            self.ffmpeg_path, "-y",
            "-i", "-",  # Read from stdin
            "-f", "mp3",
            "-b:a", self.bitrate,
            "-",        # Write to stdout
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        mp3, stderr = await ffmpeg_proc.communicate(input=audio_bytes)

        if ffmpeg_proc.returncode != 0:
            raise Exception(f"Error convert to MP3: {stderr.decode('utf-8')}")

        if verbose:
            logger.info(f"MP3 convert in {time() - start_time} sec")

        return mp3


class StyleBertVits2TTS:
    def __init__(self,
        model_dir: str = "model",
        model_file: str = "model.safetensors",
        config_file: str = "config.json",
        style_file: str = "style_vectors.npy",
        use_gpu: bool = False,
        verbose: bool = False,
        cache_backend: str = "memory",
        cache_dir: str = ".cache/tts"
    ):
        if cache_backend not in ("memory", "file"):
            raise ValueError("cache_backend must be 'memory' or 'file'")
        self.cache_backend = cache_backend
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if cache_backend == "file":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model_file = self.model_dir / model_file
        self.config_file = self.model_dir / config_file
        self.style_file = self.model_dir / style_file
        
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.cache = {}
        self.tts_model = None

        if all(f.exists() for f in [self.model_file, self.config_file, self.style_file]):
            self.load_tts_model()

    def load_tts_model(
        self, model_file: str = None,
        config_file: str = None, 
        style_file: str = None,
        use_gpu: bool = None
    ) -> bool:
        try:
            if model_file:
                self.model_file = self.model_dir / model_file
            if config_file:
                self.config_file = self.model_dir / config_file
            if style_file:
                self.style_file = self.model_dir / style_file
            if use_gpu is not None:
                self.use_gpu = use_gpu

            tts_model = TTSModel(
                model_path=str(self.model_file),
                config_path=str(self.config_file),
                style_vec_path=str(self.style_file),
                device="cuda" if self.use_gpu else "cpu"
            )
            _prepare_bert_models(tts_model.hyper_parameters.version)

            if self.cache_backend == "file":
                cache = FileCache(self.cache_dir / self._model_cache_id())
            else:
                cache = {}
            self.tts_model, self.cache = tts_model, cache

            return True

        except Exception as ex:
            logger.error(f"Error loading TTS model: {str(ex)}")
            self.tts_model = None
            return False

    def _model_cache_id(self) -> str:
        # Hash contents once per load so replacing a file at the same path is safe.
        digest = hashlib.sha256()
        for path in (self.model_file, self.config_file, self.style_file):
            file_digest = hashlib.sha256()
            with path.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    file_digest.update(chunk)
            digest.update(file_digest.digest())
        return digest.hexdigest()

    def generate_cache_key(self, text: str, speaker_id: int, style: str, additional_str: str = None, **kwargs) -> str:
        parameters = {
            "text": text,
            "speaker_id": speaker_id,
            "style": style,
            "post_processor": additional_str,
            **kwargs,
        }
        # Keep the key stable regardless of parameter order.
        serialized = json.dumps(parameters, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()

    def _converted_cache_key(self, source_key: str, post_processor: PostProcessor) -> str:
        extension = post_processor.file_extension.lstrip(".")
        if not extension or not extension.isalnum() or extension == "tmp":
            raise ValueError("PostProcessor.file_extension must be a file extension other than tmp")
        parameters = {
            "source_key": source_key,
            "post_processor": post_processor.cache_key(),
            "file_extension": extension,
        }
        serialized = json.dumps(parameters, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.md5(serialized.encode("utf-8")).hexdigest()
        return f"converted_{digest}.{extension}"

    async def _read_cache(self, cache, key: str):
        if isinstance(cache, FileCache):
            return await asyncio.to_thread(cache.get, key)
        return cache.get(key)

    async def _write_cache(self, cache, key: str, audio_data: bytes):
        if isinstance(cache, FileCache):
            await asyncio.to_thread(cache.__setitem__, key, audio_data)
        else:
            cache[key] = audio_data

    async def tts(self, text: str, speaker_id: int, style: str, post_processor: PostProcessor = None, **kwargs) -> bytes:
        # Keep an in-flight request tied to its model and cache during a reload.
        tts_model, cache = self.tts_model, self.cache
        source_hash = self.generate_cache_key(text, speaker_id, style, **kwargs)
        source_key = f"source_{source_hash}.wav"

        # A converted result can be returned without reading or generating WAV.
        if post_processor is not None:
            converted_key = self._converted_cache_key(source_key, post_processor)
            converted_audio = await self._read_cache(cache, converted_key)
            if converted_audio is not None:
                if self.verbose:
                    logger.info(f"Converted cache hit: {speaker_id}/{style}: {text}")
                return converted_audio

        audio_data = await self._read_cache(cache, source_key)
        if audio_data is None:
            # Generate and cache the source WAV independently of post processing.
            try:
                start_time = time()
                rate, audio = await asyncio.to_thread(
                    tts_model.infer,
                    text=text,
                    speaker_id=speaker_id,
                    style=style,
                    **kwargs
                )
                if self.verbose:
                    logger.info(f"Audio generated in {time() - start_time} sec: {text}")
            except Exception as ex:
                logger.error(f"Error generating audio: {str(ex)}")
                raise

            try:
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wf:
                    wf.setnchannels(1)      # mono
                    wf.setsampwidth(2)      # 16bit
                    wf.setframerate(rate)   # rate
                    wf.writeframes(audio)
                audio_data = buffer.getvalue()
            except Exception as ex:
                logger.error(f"Error converting audio to WAV: {str(ex)}")
                raise

            await self._write_cache(cache, source_key, audio_data)
        elif self.verbose:
            logger.info(f"Source cache hit: {speaker_id}/{style}: {text}")

        # No post processor means there is only one cache entry: the source WAV.
        if post_processor is None:
            return audio_data

        try:
            # The WAV header supplies the rate even when inference was skipped.
            with wave.open(io.BytesIO(audio_data), "rb") as wf:
                rate = wf.getframerate()
            converted_audio = await post_processor.process(audio_data, rate, self.verbose)
        except Exception as ex:
            logger.error(f"Error post processing audio: {str(ex)}")
            raise

        await self._write_cache(cache, converted_key, converted_audio)
        return converted_audio

    def clear_cache(self):
        self.cache.clear()
