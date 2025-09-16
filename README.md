# 小智服务器端

## 环境配置

```shell
$ conda create -n xiaozhi python=3.10
$ conda activate xiaozhi
$ pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
$ mkdir data
```

编写自定义配置 ```.config.yaml```:
```yaml
# 这是用户自定义配置文件
# 系统会优先读取此文件的配置，如果配置不存在，会自动读取根目录的config.yaml文件

# 基本配置示例（可根据需要修改）
server:
  ip: 0.0.0.0
  port: 8000
  http_port: 8003

# 如果需要修改其他配置，请参考根目录的config.yaml文件
```

### Ubuntu

```shell
$ sudo apt install opus-tools libopus-dev
# 启动服务器
$ python app.py
```

### MACOS

```shell
$ brew install opus
# 启动服务器
$ export DYLD_LIBRARY_PATH="/opt/homebrew/opt/opus/lib:$DYLD_LIBRARY_PATH" && export PKG_CONFIG_PATH="/opt/homebrew/opt/opus/lib/pkgconfig:$PKG_CONFIG_PATH" && python app.py
```

## 测试

运行起服务器后，可以通过 test 下的文件来进行测试:

```shell
$ python -m http.server 8006
```

然后在浏览器打开 ```http://localhost:8006/test_page.html``` 即可进行测试。