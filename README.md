# sbv2-api-lite

Ultra instant compatible API for Style-Bert-VITS2 - You'll have it running before your morning coffee is ready! ☕️


## 🚀 Quick start

1. Clone this repo

    ```sh
    git clone https://github.com/uezo/sbv2-api-lite
    cd sbv2-api-lite
    ```

1. Install dependencies

    ```sh
    pip install -r requirements.txt

    # If you use GPU, reinstall PyTorch that matches the CUDA version of your environment.
    pip uninstall torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124   # for CUDA 12.4
    ```

1. Start the API server

    ```sh
    uvicorn run:app --host 0.0.0.0 --port 5000
    ```


Now go to http://127.0.0.1:5000/docs . Upload yoru model file and enjoy high quality TTS🥳



## Language models

Loading a voice model also prepares its BERT model and tokenizer resources:

- JP-Extra models prepare Japanese resources only.
- Normal models prepare Japanese, English, and Chinese resources. Their
  configuration does not identify a single language to use.

Missing resources are downloaded automatically from Hugging Face. Downloaded
files are reused from its cache on later starts, and loaded resources are shared
between voice models within the same process. These assets use the Hugging Face
cache, separately from the generated-audio cache below.

The first load can take longer and requires internet access to download missing
files. Normal models also require memory for the additional language resources.
If preparing a required resource fails, the voice model load fails. Importing
the module or starting without a voice model does not download these resources.
No changes to `run.py` are needed to enable this behavior.


## Audio cache

The cache backend is selected at startup. By default, generated audio is cached
in memory and discarded when the process exits.

| Environment variable | Default | Description |
| --- | --- | --- |
| `SBV2_API_LITE_CACHE_BACKEND` | `memory` | `memory` or `file` |
| `SBV2_API_LITE_CACHE_DIR` | `.cache/tts` | Root directory for the file cache |

To persist generated audio on the filesystem:

```sh
SBV2_API_LITE_CACHE_BACKEND=file \
SBV2_API_LITE_CACHE_DIR=/path/to/cache \
uvicorn run:app --host 0.0.0.0 --port 5000
```

Python callers can pass `cache_backend="file"` and `cache_dir="/path/to/cache"`
to `StyleBertVits2TTS`.

Both backends cache audio in two stages:

- The **source WAV** is keyed by all synthesis parameters, including `language`.
  Without a post processor, this is the only stored entry.
- A **converted result** is keyed by the source key and the post processor's
  identity, settings, and output extension. A hit returns it directly. Otherwise,
  the source WAV is reused for conversion, so changing the output format or
  bitrate does not repeat inference.

Files retain their audio extensions and can be opened directly:

```text
<cache_dir>/<model_hash>/source_<synthesis_hash>.wav
<cache_dir>/<model_hash>/converted_<conversion_hash>.mp3
```

The model hash is calculated from the contents of the model, configuration, and
style-vector files once when the model is loaded. Changing any of these files
selects a different cache directory, even if the filename stays the same. Loading
the same model again, including after a restart, reuses its saved audio. Memory
caches are reset on every model load.

Custom `PostProcessor` subclasses implement
`async process(self, audio_bytes, sample_rate, verbose=False) -> bytes` and declare
`file_extension`, such as `"wav"`, `"ogg"`, or `"flac"` (default: `"bin"` when
unspecified). Override `cache_key()` to include all settings that affect the
output. The built-in MP3 processor includes its bitrate and ffmpeg path.
WAV-to-WAV processing is supported: the source and converted `.wav` files have
different keys and are both preserved.

To clear the current model's source WAV and converted audio caches:

```sh
curl -X DELETE http://127.0.0.1:5000/cache
```

The endpoint returns HTTP 200 with `{"result":"success"}`, including when the
cache is already empty. It supports both memory and file backends. With a router
prefix such as `/tsumugi`, use `/tsumugi/cache` to clear that model's cache.

Python callers can use `clear_cache()` for the same operation. Older models'
file caches remain on disk until explicitly removed. Memory caches are cleared
only in the process handling the request. Neither backend has an expiration time
or size limit.


## 📦　Run on RunPod

Set the following command as the Container Start Command in your Pod console, then start the Pod to execute the process. Once complete, stop the Pod in the same order.

1. Set up the repository in the workspace

    ```sh
    git clone https://github.com/uezo/sbv2-api-lite /workspace/sbv2-api-lite
    ```

1. Create the runtime environment and install dependencies

    ```sh
    bash /workspace/sbv2-api-lite/runpod_install.sh
    ```

1. Set up the API server to launch at startup

    ```sh
    bash /workspace/sbv2-api-lite/runpod_start.sh
    ```

    Add the following environment variables with the value `true`:

    ```
    SBV2_API_LITE_USE_GPU
    SBV2_API_LITE_VERBOSE
    ```

    If you want to use compressed audio, add the variable with the value `ffmpeg-static/ffmpeg`:

    ```
    SBV2_API_LITE_FFMPEG_PATH
    ```

1. Click **Connect to HTTP Service [Port 5000]**

    Click **Connect to HTTP Service [Port 5000]** button and head over to /docs to see the OpenAPI interface.


Upload model file with **GPU=True**. Enjoy high quality TTS🥳

Note: You only need to configure this setup once. After that, just start the Pod, and the API server will automatically launch.


