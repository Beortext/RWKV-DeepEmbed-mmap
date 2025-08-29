import torch
import struct
import json
import os
import io
from tqdm import tqdm

# (您的 get_DeepEmbed 函数保持不变, 此处省略)
key_words = {"s_emb", "k_emb", "v_emb"}
def get_DeepEmbed(model_name: str):
    # ... (代码与您提供的一致)
    pth = torch.load(model_name if model_name.endswith(".pth") else f"{model_name}.pth", map_location="cpu", weights_only=True)
    n_layer = max(int(k.split(".")[1]) for k in pth.keys() if any(kw in k for kw in key_words)) + 1
    print(f"Layer number: {n_layer}")
    de_pth = {}
    norm_emb = torch.nn.functional.layer_norm(pth['emb.weight'], (pth['emb.weight'].shape[-1],), weight=pth['blocks.0.ln0.weight'], bias=pth['blocks.0.ln0.bias'])
    for i in range(n_layer):
        de_pth[f"s_emb.{i}"] = pth[f"blocks.{i}.ffn.s_emb.weight"] + norm_emb @ pth[f"blocks.{i}.ffn.s_emb_x.weight"].t()
        de_pth[f"k_emb.{i}"] = pth[f"blocks.{i}.qkv.k_emb.weight"] + norm_emb @ pth[f"blocks.{i}.qkv.k_emb_x.weight"].t()
        de_pth[f"v_emb.{i}"] = pth[f"blocks.{i}.qkv.v_emb.weight"] + norm_emb @ pth[f"blocks.{i}.qkv.v_emb_x.weight"].t()
    print("DeepEmbed(A) build OK.")
    return de_pth


def create_data_store_optimized(output_path, original_data_dict):
    """
    优化版：采用 Data | Index | Footer 结构，单遍写入，逻辑更清晰。
    Footer格式: [index_offset (uint64), index_size (uint64)]
    """
    if os.path.exists(output_path):
        print(f"文件 {output_path} 已存在，将被覆盖。")
        os.remove(output_path)
    
    dtype_map = {
        torch.float32: 0, torch.float16: 1, torch.bfloat16: 2,
        torch.int64: 3, torch.int32: 4, torch.uint8: 5
    }

    final_index = {}
    
    with open(output_path, 'wb') as f:
        # --- 步骤 1: 依次写入所有Tensor数据 ---
        print("[1/3] Writing tensor data blocks...")
        for key, tensor in tqdm(original_data_dict.items()):
            tensor_u8 = tensor.contiguous().view(torch.uint8)
            tensor_bytes = tensor_u8.numpy().tobytes()
            
            final_index[key] = {
                "offset": f.tell(), # 直接记录当前位置作为绝对偏移
                "shape": list(tensor.shape),
                "dtype": dtype_map[tensor.dtype]
            }
            f.write(tensor_bytes)

        # --- 步骤 2: 写入索引区 ---
        print("[2/3] Writing JSON index...")
        index_offset = f.tell() # 数据区结束的位置就是索引区的开始
        index_bytes = json.dumps(final_index).encode('utf-8')
        f.write(index_bytes)
        index_size = len(index_bytes)

        # --- 步骤 3: 在文件末尾写入包含元信息的Footer ---
        print("[3/3] Writing footer...")
        f.write(struct.pack('<QQ', index_offset, index_size))

    print(f"数据文件 {output_path} 构建成功!")
    print(f"Index starts at {index_offset}, size is {index_size}.")

# --- 使用示例 ---
if __name__ == '__main__':
    model_weights = get_DeepEmbed("rwkv7b-g1b-0.1b-20250822-ctx4096.pth")
    create_data_store_optimized("DeepEmbed.bin", model_weights)