import asyncio
import hashlib
import io
import logging
from pathlib import Path
from time import time
import wave
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel


# Get logger
logger = logging.getLogger(__name__)


# Import base model
logger.info("Start loading base model. This may take several minutes...")
bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
logger.info("Finish loading base model.")


class StyleBertVits2TTS:
    def __init__(self,
        model_dir: str = "model",
        model_file: str = "model.safetensors",
        config_file: str = "config.json",
        style_file: str = "style_vectors.npy",
        use_gpu: bool = False,
        verbose: bool = False
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model_file = self.model_dir / model_file
        self.config_file = self.model_dir / config_file
        self.style_file = self.model_dir / style_file
        
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.cache = {}  # Simple dictionary cache
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

            self.tts_model = TTSModel(
                model_path=str(self.model_file),
                config_path=str(self.config_file),
                style_vec_path=str(self.style_file),
                device="cuda" if self.use_gpu else "cpu"
            )

            self.clear_cache()

            return True

        except Exception as ex:
            logger.error(f"Error loading TTS model: {str(ex)}")
            self.tts_model = None
            return False

    def generate_cache_key(self, text: str, speaker_id: int, style: str) -> str:
        unique_string = f"{text}_{speaker_id}_{style}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    async def tts(self, text: str, speaker_id: int, style: str, **kwargs) -> bytes:
        cache_key = self.generate_cache_key(text, speaker_id, style)

        # Check cache
        if cache_key in self.cache:
            if self.verbose:
                logger.info(f"Cache hit: {speaker_id}/{style}: {text}")
            return self.cache[cache_key]

        # Run the TTS in a separate thread
        try:
            start_time = time()
            rate, audio = await asyncio.to_thread(
                self.tts_model.infer,
                text=text,
                speaker_id=speaker_id,
                style=style,
                **kwargs
            )
            if self.verbose:
                logger.info(f"Audio generated in {time() - start_time} sec: {text}")

        except Exception as ex:
            logger.error(f"Error generating audio: {str(ex)}")
            raise ex

        # Convert to WAV
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)      # mono
                wf.setsampwidth(2)      # 16bit
                wf.setframerate(rate)   # rate
                wf.writeframes(audio)
            buffer.seek(0)

            # Store in cache
            audio_data = buffer.read()
            self.cache[cache_key] = audio_data
            return audio_data

        except Exception as ex:
            logger.error(f"Error converting audio to WAV: {str(ex)}")
            raise ex

    def clear_cache(self):
        self.cache.clear()
