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


