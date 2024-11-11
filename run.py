import os
def str_to_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "y", "yes")

USE_GPU = str_to_bool(os.getenv("SBV2_API_LITE_USE_GPU", "false"))
VERBOSE = str_to_bool(os.getenv("SBV2_API_LITE_VERBOSE", "false"))
LOG_LEVEL = os.getenv("SBV2_API_LITE_LOG_LEVEL", "INFO").upper()
FFMPEG_PATH = os.getenv("SBV2_API_LITE_FFMPEG_PATH", "ffmpeg")
MP3_BITRATE = os.getenv("SBV2_API_LITE_MP3_BITRATE", "64k")


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

from fastapi import FastAPI
from sbv2apilite.tts import StyleBertVits2TTS
from sbv2apilite.api import get_api_router


sbv2tts = StyleBertVits2TTS(use_gpu=USE_GPU, verbose=VERBOSE)
router = get_api_router(sbv2tts, MP3_BITRATE, FFMPEG_PATH)

app = FastAPI(title="Style-Bert-VITS2 API Lite")
app.include_router(router)
