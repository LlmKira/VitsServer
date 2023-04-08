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

我们提供开箱即用的调用系统。

- [Python SDK](docs/sdk.py)
- [JavaScript SDK](docs/sdk.js)

