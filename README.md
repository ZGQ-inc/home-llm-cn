> [!CAUTION]
> WIP.

# Home LLM CN

Home LLM 中文版。

原仓库：https://github.com/acon96/home-llm

# 训练方法

克隆此仓库：

```
git clone https://github.com/ZGQ-inc/home-llm-cn.git
```

进入 train 文件夹。

## 环境配置

### Python

安装Python：https://www.python.org/downloads/

创建并激活虚拟环境：

```
python -m venv .venv
```

```
.\venv\Scripts\activate
```

### Pytorch

安装 CUDA Toolkit：https://developer.nvidia.com/cuda/toolkit

安装 Pytorch 12.8 (这里使用 12.8 版本是为了适配最新的 Blackwell 架构的N卡，通常 PyTorch 目前更稳定的是 12.4，如果你使用老架构，请安装 `cu124`)：

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

安装训练需要的依赖：

```
pip install transformers peft datasets trl bitsandbytes scipy accelerate jinja2 huggingface_hub
```

### Git

安装 Git 、Git LFS 和 `git-xet`：

https://git-scm.com/install/

https://git-lfs.com/

```
winget install git-xet
```

### Llama

克隆 llama.cpp 仓库：

```
git clone https://github.com/ggml-org/llama.cpp.git
```

安装依赖：

```
pip install -r llama.cpp/requirements.txt
```

如果碰到依赖问题，请使用此仓库内的文件：[./train/llama.cpp/requirements](./train/llama.cpp/requirements)

安装 Ollama：https://ollama.com/download

下载 llama.cpp 的 release：https://github.com/ggml-org/llama.cpp/releases

推荐 `llama-bxxxx-bin-win-cpu-x64.zip`，解压。

## 下载模型和数据集

### gemma-3-1b-it

打开模型仓库：https://huggingface.co/google/gemma-3-1b-it ，点击 `Agree and access repository` 同意许可。

申请 `Read` 类型 token：https://huggingface.co/settings/tokens

克隆模型仓库：

```
git clone https://<用户名>:<token>@huggingface.co/google/gemma-3-1b-it
```

### translategemma:12b

```
ollama pull translategemma:12b
```

### home_assistant_train_english.jsonl

下载数据集，放进 datasets 文件夹：https://huggingface.co/datasets/acon96/Home-Assistant-Requests-V2/resolve/main/home_assistant_train_english.jsonl

## 翻译数据集

原始数据集为英文，需要使用 translategemma 进行翻译。

进入 datasets 文件夹，运行 translategemma：

```
ollama run translategemma:12b
```

Ctrl + D 退出，依次运行 提取文本、翻译和替换步骤：

```
python pick.py
```

```
python translate.py
```

```
python rp.py
```

把翻译好的 `home_assistant_train_chinese.jsonl` 移动到 train 文件夹。

## 开始训练

### 训练

```
python train.py
```

测试LoRA（可选）：

```
python testlora.py
```

### 合并

```
python merge.py
```

### 转换

```
python llama.cpp/convert_hf_to_gguf.py ./gemma-ha-1b-merged --outfile name.gguf --outtype f16
```

### 量化（可选）

查看可用量化类型：

```
./llama-b7999-bin-win-cpu-x64\llama-quantize.exe --help
```

量化：

```
./llama-b7999-bin-win-cpu-x64\llama-quantize.exe name.gguf name_<量化类型>.gguf <量化类型>
```

### 导入

使用 [./convert](./convert) 里面的模板文件：

```
ollama create <name> -f Modelfile_template
```
