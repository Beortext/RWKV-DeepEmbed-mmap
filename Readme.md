# RWKV DeepEmbed mmap 高效推理方案

本项目实现了一种针对 RWKV 模型中 DeepEmbed 部分的高效加载和推理方案。通过将 DeepEmbed 权重预处理成二进制文件，并利用内存映射（mmap）和异步加载技术，显著降低了模型加载时的内存峰值和IO开销，优化了推理性能。

此方案的核心思想是将原本在模型加载时需要通过大量torch张量计算才能得到的DeepEmbed，转化为一个可以直接读取的二进制数据文件 (`.bin`)。

## ✨ 项目特点

* **低内存峰值**：使用内存映射（mmap）加载DeepEmbed，避免了在加载模型时一次性将全部权重读入内存，从而大大降低了内存占用的峰值。
* **快速加载**：二进制文件配合异步加载器 (`async_loader.cpp`)，使得 DeepEmbed 的读取过程更加高效。
* **优化IO**：将多个离散的权重整合为单个二进制索引，减少了文件句柄的开销和磁盘寻址次数。

## 🛠️ 工作流程

1. **预处理 (`create_de_bin.py`)**:

      * 加载原始的 `.pth` 模型文件。
      * (可选) 生成一个不包含 DeepEmbed 权重的“主模型”文件 (`*_NoDE.pth`)。
      * 计算出所有层级的 DeepEmbed (`s_emb`, `k_emb`, `v_emb`)。
      * 将计算好的 DeepEmbed 权重序列化为一个自定义格式的二进制文件 (`.bin`)。该文件内部包含数据区、JSON索引区和元数据Footer。

2. **推理 (`rwkv7b_demo_v3.py`)**:

      * 加载不含 DeepEmbed 的主模型。
      * 通过C++实现的异步加载器 `async_loader` 读取 `.bin` 文件中的DeepEmbed权重。
      * 在推理过程中，根据输入的 `token`，异步地从 `.bin` 文件中获取对应的 `s_emb`, `k_emb`, `v_emb` 并加载到GPU。
      * 执行完整的模型推理。

## ⚙️ 如何使用

### 1\. 环境准备

您需要安装 PyTorch 以及支持编译C++/CUDA的环境。

```bash
# 示例环境，请根据您的实际情况调整
pip install torch tqdm numpy
```

### 2\. 生成 DeepEmbed 二进制文件

使用 `create_de_bin.py` 脚本来处理原始的 RWKV 模型文件。

**命令格式**：

```bash
python create_de_bin.py <模型路径.pth> -o <输出文件名.bin> [--no-gen-main]
```

**参数说明**：

* `<模型路径.pth>`: 必需，指向原始 RWKV 模型的 `.pth` 文件路径。
* `-o, --output`: 可选，指定输出的二进制文件名，默认为 `DeepEmbed.bin`。
* `--no-gen-main`: 可选，如果设置此项，将不会生成 `*_NoDE.pth` 主模型文件。

**示例**：

```bash
python create_de_bin.py rwkv-7b-world.pth -o DeepEmbed.bin
```

运行后，您将得到 `DeepEmbed.bin` 和 `rwkv-7b-world_NoDE.pth` 两个文件。

### 3\. 运行推理Demo

编译C++/CUDA算子并运行推理脚本。

首先，确保 `rwkv7b_demo_v3.py` 中的模型路径、词表路径和 `DeepEmbed.bin` 路径配置正确。

然后执行脚本：

```bash
python rwkv7b_demo_v3.py
```

脚本会自动调用 `torch.utils.cpp_extension.load` 来编译所需的C++和CUDA代码。

## 📄 测试

项目内包含一个测试脚本 `mmap_de_test.py`，可用于验证生成的 `.bin` 文件的正确性。它会随机抽样检查从二进制文件中读取的张量与原始计算的张量是否一致。
