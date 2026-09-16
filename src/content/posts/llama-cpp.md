---
title: "llama.cpp"
pubDate: 2025-07-10
description: "记录 llama.cpp 的构建、GGUF 转换、量化、推理和镜像构建。"
author: "Westwoods"
tags: [llama.cpp, LLM, 部署]
draft: false
---

`llama.cpp` 其实是一套工具，其将大模型编译为 C/C++来实现高效推理 [ggml-org/llama.cpp: LLM inference in C/C++](https://github.com/ggml-org/llama.cpp)。具体来说，其支持这些功能：
1. 线下模型聊天
2. llama-server：Start a local HTTP server with default configuration on port 8080，同时支持文本聊天、多模态、词嵌入、重排序模型
3. 模型复杂度评估
4. 测试模型
5. 量化：压缩模型的精度，减小模型体积，改善效率


`llama.cpp` 编译得到的模型几乎支持所有的硬件平台，尤其是苹果 m 系列芯片。而且处理之后的模型运行效率极高，还允许使用显存和内存同时推理，不可谓不强大。Ollama 其实就是打包了一下。
# 使用
## 构建
部署 llama. cpp 需要先克隆他们的项目，然后用 cmake 编译。make 的编译方案已经过时了，那些让你用 make 编译的教程通通不能用。官网的构建方法：
```bash
cmake -B build
cmake --build build --config Release -j 16 # 多线程
```

构建之后就可以直接运行 GGUF 大模型了。

如果不需要自动下载-构建，只需要量化下载到本地的模型，可以直接使用：
```bash
cmake -B build -DLLAMA_CURL=OFF # 禁用网络下载器
```

除了 CPU 构建，也可以用 BLAS 构建。这会需要 OpenBLAS 支持。：
```bash
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DLLAMA_CURL=OFF # 禁用网络下载器
cmake --build build --config Release -j 16 # 多线程
```

使用 BLAS 构建不会影响推理速度。但是可以加速大批量推理（多于512）中的提示词处理速度。

如果要使用 GPU 实现各种加速，就需要在 cmake 和 gcc 的基础上使用 cuda 以及 nvcc。像是 cuda 和 nvcc、gcc 这类工具都可以在标签中含有 `devel` 的 cuda 镜像中找到。

用 cuda 编译需要需要在参数中加入 ` -DGGML_CUDA=ON `：

```bash
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF # 禁用网络下载器
cmake --build build --config Release -j 16 # 多线程
```

构建完之后，就可以在 `build/bin` 目录下找到全套工具了。

> [!tip]
> 执行构建失败之后，必须删除 build 文件清理缓存，否则导致失败的配置不会被更新。
## 转换 GGUF
在量化之前，需要把 safe_tensor 这样的格式转换为 GGUF。具体参见 HF转GGUF。

如果要把其他模型转换为 GGUF ，则需要使用对应的 `convert_*.py` 脚本。让你用 `convert.py` 的教程也全都过时了。如今提供的工具是这些：
![](/blog-assets/llama.cpp-1751431757967.png)

HuggingFace 或者 ModelScope 下载的模型都使用 `convert_hf_to_gguf.py` 进行转换：
```bash
python convert_hf_to_gguf.py /workspace/Qwen2.5-32B-Instruct/ --outfile qwen2.5-32B-Instruct-FP16.gguf --outtype f16
```
这一步无需进行精度转换，统一使用原始精度（f16 或 f32）即可。压缩模型会在之后的量化阶段完成。

可以使用 `python convert_hf_to_gguf.py -h` 获取帮助。

具体参照 HF转GGUF。
> [!tip]
> 理论上来说使用这个脚本不需要安装特别的 python 库。但是如果它提示缺库的话单独下载缺失的库即可。

## 量化
量化（quantize）指的是修改模型权重的数据表示。比如 32 位浮点数压缩为 8 位甚至是 4 位、2 位。压缩之后模型体积会减小，推理效率会提高，但是相应的推理效果会有一定衰退。

使用 llama.cpp 量化需要使用构建出来的可执行文件，位于 `build/bin`。一般教程里面会说使用 `quantize` 来进行量化，这是过时的版本，准确的来说要使用 `llama-quantize` 。
```bash
llama-quantize Yi-34B.gguf Yi-34B-Q6_K.gguf Q6_K 16
```
## 测试

## 执行推理
```bash
llama-server -m qwen2.5-32B-Instruct-Q6_K.gguf -ngl 999 -ts 0.5,0.5 -c 4096 --host 0.0.0.0 --port 11434
```

```
docker run -it --name LLMs_Workshop -v /mnt/data/fanghejin/LLMs_Workshop:/workspace --gpus all -p 12434:11434  llms-workshop:0.1.0
```
# 镜像构建
```dockerfile
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# -----------------------------
# Step 1: Set APT mirror and install dependencies
# -----------------------------
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    git \
    cmake \    
    libcurl4-openssl-dev \
    curl \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------
# Step 2: Configure pip mirror globally
# -----------------------------
RUN mkdir -p /etc/pip && \
    echo '[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple' > /etc/pip.conf && \
    pip3 install --upgrade pip setuptools wheel

# -----------------------------
# Step 3: Copy llama.cpp source and build
# -----------------------------
COPY llama.cpp /opt/llama.cpp
WORKDIR /opt/llama.cpp

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so && \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 &&\
    rm -r build && \
    cmake -B build -DGGML_CUDA=ON && \
    cmake --build build --config Release -j $(nproc) && \
    cp build/bin/llama-* build/bin/test-* /usr/local/bin/ && \
    cp build/bin/lib*.so /usr/local/lib/ && \
    chmod +x /usr/local/bin/llama-* /usr/local/bin/test-* && \
    ldconfig

# -----------------------------
# Step 4: Install Python requirements for GGUF tools
# -----------------------------
RUN pip3 install -r requirements.txt

# -----------------------------
# Step 5: Set working directory and shell
# -----------------------------
RUN mkdir -p /workspace
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]

```
