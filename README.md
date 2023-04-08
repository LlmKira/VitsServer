# Vits-Server 🔥

⚡ A VITS ONNX server designed for fast inference, supporting streaming and additional inference settings to enable model
preference settings and optimize performance.

[![Docker](https://github.com/LlmKira/VitsServer/actions/workflows/docker-latest.yaml/badge.svg)](https://github.com/LlmKira/VitsServer/actions/workflows/docker-latest.yaml)

## Advantages 💪

- Automatic language type parsing for text, eliminating the need for language recognition segmentation.
- Supports multiple audio formats, including ogg, wav, flac, and silk.
- Multiple models, streaming inference.
- Additional inference settings to enable model preference settings and optimize performance.

## API Documentation 📖

We offer out-of-the-box call systems.

- [Python SDK](docs/sdk.py)
- [JavaScript SDK](docs/sdk.js)

```python
client = VITS("http://127.0.0.1:9557")
res = client.generate_voice(model_id="model_01", text="你好，世界！", speaker_id=0, audio_type="wav",
                            length_scale=1.0, noise_scale=0.5, noise_scale_w=0.5, auto_parse=True)
with open("output.wav", "wb") as f:
    for chunk in res.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)
```

## Running 🏃

We recommend using a virtual environment to isolate the runtime environment. Because this project's dependencies may
potentially disrupt your dependency library, we recommend using `pipenv` to manage the dependency package.

### Testing from Shell 🐚

Configuration is in config.toml, including the following fields:

```toml
[server]
host = '0.0.0.0'
port = 9557
reload = false
```

```shell
apt install python3-pip
pip3 install pipenv
pipenv install    # Create and install dependency packages
pipenv shell      # Activate the virtual environment
python3 server.py # Run

```

### Running from pm2.json 🚀

```shell
apt-get update &&
  apt-get install -y build-essential libsndfile1 vim gcc g++ cmake &&
  python3 -m pip install -r requirements.txt
apt install npm
npm install pm2 -g
pm2 start pm2.json

```

### Building from Docker 🐋

```shell
docker build -t <image-name> .
```

where `<image-name>` is the name you want to give to the image. Then, use the following command to start the container:

```shell
docker run -it -p 9557:9557 -v <local-path>/vits_model:/app/model <image-name>
```

where `<local-path>` is the local folder path you want to map to the /app/model directory in the container.

## Model Configuration 📁

In the `model` folder, place the `model.pth`/ `model.onnx` and corresponding `model.json` files. If it is `.pth`, it
will be automatically converted to `.onnx`!

### Model Extension Design 🔍

You can add extra fields in the model configuration to obtain information such as the model name corresponding to the
model ID through the API.

```json5
{
  //...
  "info": {
    "name": "coco",
    "description": "a vits model",
    "author": "someone",
    "cover": "https://xxx.com/xxx.jpg",
    "email": "xx@ws.com"
  },
  "infer": {
    "noise_scale": 0.667,
    "length_scale": 1.0,
    "noise_scale_w": 0.8
  }
  //....
}
```

### How can I retrieve model information?

You can access f"{self.base_url}/model/list?show_speaker=True&show_ms_config=True" to obtain detailed information about
model roles and configurations.