# Vits-Server

## 优势

- 自动解析文本的语言类型，无需处理语言识别切分。
- 支持多种音频格式，包括ogg、wav、flac、silk。
- 多模型，流式传输。
- 额外推理设置，为模型启用偏好设置，提升效果。

## 运行

推荐使用虚拟环境隔离运行环境。因为本项目依赖有可能破坏您的依赖库，所以推荐使用 `pipenv` 来管理依赖包。

### 从 Shell 测试

配置在 config.toml 中，包括以下字段：

```toml
[server]
host = '0.0.0.0'
port = 9557
reload = false
```

```shell
apt install python3-pip
pip3 install pipenv
pipenv install           # 创建并安装依赖包
pipenv shell             # 激活虚拟环境
python3 server.py          # 运行
```

### 从 pm2.json 运行

```shell
apt install npm
npm install pm2 -g
pm2 start pm2.json
```

### 从 Docker 构建

```shell
docker build -t <镜像名称> .
```

其中 <镜像名称> 是您想要给该镜像命名的名称。然后，使用以下命令将容器启动起来：

```shell
docker run -it -p 9557:9557 -v <本地路径>/vits_model:/app/model <镜像名称>
```

其中 <本地路径> 是您想要映射到容器中 /app/model 目录的本地文件夹路径。

## 模型配置

在 `model` 文件夹下，放入 `model.pth`/ `model.onnx` 和对应的 `model.json` 文件即可。如果是 `.pth` ，会自动转换为 `.onnx`！

### 模型扩展设计

你可以在模型配置中加入额外的字段，通过Api获取模型ID对应的模型名称等信息。

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

## API文档

### 获取模型列表和信息

- URL: /model/list
- Method: GET
- Request Parameters:
    - show_speaker (optional): 是否显示说话人列表，bool类型，默认为False。
    - show_ms_config (optional): 是否显示模型配置参数，bool类型，默认为False。
- Response:
    - code: 响应状态码，int类型。
    - message: 响应消息，str类型。
    - data: 响应数据，list类型。每个元素包含以下字段：
        - model_id: 模型ID，str类型。
        - model_info: 模型信息，dict类型。
        - speaker (optional): 说话人列表，list类型。仅在show_speaker为True时出现。
        - speaker_num (optional): 说话人数量，int类型。仅在show_speaker为True时出现。
        - ms_config (optional): 模型配置参数，dict类型。仅在show_ms_config为True时出现。

```shell
curl -X 'GET' \
  'http://127.0.0.1:9557/model/list?show_speaker=false&show_ms_config=false' \
  -H 'accept: application/json'
```

```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "model_id": "374_epochs.pth",
      "model_info": {
        "name": "name",
        "description": "name",
        "author": "name",
        "cover": "",
        "email": ""
      }
    }
  ]
}
```

### 获取模型名称对应的设置参数

- URL: /model/info
- Method: GET
- Request Parameters:
    - model_id: 模型ID，str类型。
- Response:
    - code: 响应状态码，int类型。
    - message: 响应消息，str类型。
    - data: 响应数据，dict类型。表示该模型的设置参数。

```json5
// model config json file content
```

### 处理传入文本为Vits格式包装

- URL: /tts/parse
- Method: POST
- Request Parameters:
    - text: 待处理的文本，str类型。
    - strip (optional): 是否去除空格，bool类型，默认为False。
- Response:
    - code: 响应状态码，int类型。
    - message: 响应消息，str类型。
    - data: 响应数据，dict类型。包含以下字段：
        - detect_code: 检测到的语言编码，str类型。
        - parse: 处理后的文本信息，dict类型。
        - raw_text: 原始文本，str类型。
        - result: Vits格式的文本，str类型。

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "detect_code": false,
    "parse": [
      {
        "text": "今天是个晴朗的日子，阳光明媚，空气清新。我打算去公园散步，享受这美好的一天,这句话的翻译是:",
        "lang": "ZH"
      },
      {
        "text": " 今日は晴天で、日差しがまぶしいです。新鮮な空気が流れています。私は公園に散歩に行くつもりで、この素晴らしい日を.",
        "lang": "JA"
      },
      {
        "text": "     무엇인가를 생각하면 답답하거나 짜증나지 않고 미소 머금을 수 있는 하루였으면 좋겠습니다.",
        "lang": "KO"
      },
      {
        "text": " I hope that I can smile with a smile without being annoyed or annoyed when I think of something.",
        "lang": "EN"
      },
      {
        "text": "",
        "lang": ""
      }
    ],
    "raw_text": "今天是个晴朗的日子，阳光明媚，空气清新。我打算去公园散步，享受这美好的一天,这句话的翻译是: 今日は晴天で、日差しがまぶしいです。新鮮な空気が流れています。私は公園に散歩に行くつもりで、この素晴らしい日を.     무엇인가를 생각하면 답답하거나 짜증나지 않고 미소 머금을 수 있는 하루였으면 좋겠습니다. I hope that I can smile with a smile without being annoyed or annoyed when I think of something.",
    "result": "[ZH]今天是个晴朗的日子，阳光明媚，空气清新。我打算去公园散步，享受这美好的一天,这句话的翻译是:[ZH] [JA] 今日は晴天で、日差しがまぶしいです。新鮮な空気が流れています。私は公園に散歩に行くつもりで、この素晴らしい日を.[JA] [KO]     무엇인가를 생각하면 답답하거나 짜증나지 않고 미소 머금을 수 있는 하루였으면 좋겠습니다.[KO] [EN] I hope that I can smile with a smile without being annoyed or annoyed when I think of something.[EN]"
  }
}
```

### TTS语音合成

- URL: /tts/generate
- Method: POST
- Request Body:
    - tts_req: TtsSchema类型。请求参数包括以下字段：
        - model_id: 模型ID，str类型。
        - text: 待转换的文本，str类型。默认值为"今天的天气真好！Do You Think So? 日语：今日は天気がいいですね！"。
        - speaker_id (optional): 说话人ID，int类型。默认为0。
        - audio_type (optional): 音频类型，str类型。取值为["ogg", "wav", "flac", "silk"]之一，默认为"wav"。
        - length_scale (optional): 调整语速的倍数，float类型。默认为1.0。
        - noise_scale (optional): 增加噪声的强度，float类型。默认为0.667。
        - noise_scale_w (optional): 增加噪声的类型，float类型。默认为0.8。
        - sample_rate (optional): 音频采样率，int类型。默认为22050。
        - load_prefer (optional): GPU/CPU优化选项，bool类型。默认为True。
    - auto_parse (optional): 是否自动解析Vits格式的文本，bool类型。默认为False。
- Response:
    - StreamingResponse类型。返回生成的音频文件流，media_type为"application/octet-stream"。

## Python调用示例

### 获取模型列表和信息

```python
import requests

url = "http://localhost/model/list"
params = {"show_speaker": True, "show_ms_config": True}
response = requests.get(url, params=params)
if response.status_code == 200:
    data = response.json()["data"]
    for item in data:
        print(f"model_id: {item['model_id']}")
        print(f"model_info: {item['model_info']}")
        if "speaker" in item:
            print(f"speaker: {item['speaker']}")
            print(f"speaker_num: {item['speaker_num']}")
        if "ms_config" in item:
            print(f"ms_config: {item['ms_config']}")
else:
    print("Request Failed.")
```

### 获取模型名称对应的设置参数

```python
import requests

url = "http://localhost/model/info"
params = {"model_id": "tts_model_001"}
response = requests.get(url, params=params)
if response.status_code == 200:
    data = response.json()["data"]
    print(data)
else:
    print("Request Failed.")
```

### 处理传入文本为Vits格式包装

```python
import requests

url = "http://localhost/tts/parse"
data = {"text": "这是一段待处理的文本。", "strip": True}
response = requests.post(url, json=data)
if response.status_code == 200:
    result = response.json()["data"]
    print(f"detect_code: {result['detect_code']}")
    print(f"parse: {result['parse']}")
    print(f"raw_text: {result['raw_text']}")
    print(f"result: {result['result']}")
else:
    print("Request Failed.")
```

### TTS语音合成

```python
import requests

url = "http://localhost/tts/generate"
data = {
    "model_id": "tts_model_001",
    "text": "这是一段待转换的文本。",
    "speaker_id": 0,
    "audio_type": "wav",
    "length_scale": 1.0,
    "noise_scale": 0.667,
    "noise_scale_w": 0.8,
    "sample_rate": 22050,
    "load_prefer": True
}
response = requests.post(url, json=data, stream=True)
if response.status_code == 200:
    with open("output.wav", "wb") as f:
        for chunk in response.iter_content(chunk_size=None):
            f.write(chunk)
else:
    print("Request Failed.")
```

## JavaScript调用示例

### 获取模型列表和信息

```javascript
const params = new URLSearchParams({
    show_speaker: true,
    show_ms_config: true
});
fetch("http://localhost/model/list?" + params.toString())
    .then(response => response.json())
    .then(data => {
        for (let item of data.data) {
            console.log(`model_id: ${item.model_id}`);
            console.log(`model_info: ${JSON.stringify(item.model_info)}`);
            if ("speaker" in item) {
                console.log(`speaker: ${JSON.stringify(item.speaker)}`);
                console.log(`speaker_num: ${item.speaker_num}`);
            }
            if ("ms_config" in item) {
                console.log(`ms_config: ${JSON.stringify(item.ms_config)}`);
            }
        }
    })
    .catch(error => console.error(error));
```

### 获取模型名称对应的设置参数

```javascript
const params = new URLSearchParams({
    model_id: "tts_model_001"
});
fetch("http://localhost/model/info?" + params.toString())
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));
```

### 处理传入文本为Vits格式包装

```javascript
const data = {
    text: "这是一段待处理的文本。",
    strip: true
};
fetch("http://localhost/tts/parse", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
})
    .then(response => response.json())
    .then(data => {
        const result = data.data;
        console.log(`detect_code: ${result.detect_code}`);
        console.log(`parse: ${JSON.stringify(result.parse)}`);
        console.log(`raw_text: ${result.raw_text}`);
        console.log(`result: ${result.result}`);
    })
    .catch(error => console.error(error));
```

### TTS语音合成

```javascript
const data = {
    model_id: "tts_model_001",
    text: "这是一段待转换的文本。",
    speaker_id: 0,
    audio_type: "wav",
    length_scale: 1.0,
    noise_scale: 0.667,
    noise_scale_w: 0.8,
    sample_rate: 22050,
    load_prefer: true
};
fetch("http://localhost/tts/generate", {
    method: "POST",
    body: JSON.stringify(data),
    headers: {"Content-Type": "application/json"}
})
    .then(response => {
        const reader = response.body.getReader();
        const stream = new ReadableStream({
            start(controller) {
                function push() {
                    reader.read().then(({done, value}) => {
                        if (done) {
                            controller.close();
                            return;
                        }
                        controller.enqueue(value);
                        push();
                    });
                }

                push();
            }
        });
        return new Response(stream, {
            headers: {"Content-Type": "application/octet-stream"}
        });
    })
    .then(response => response.blob())
    .then(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "output.wav";
        a.click();
        URL.revokeObjectURL(url);
    })
    .catch(error => console.error(error));
```