import torch, struct, json, os, argparse
from tqdm import tqdm


key_words = {"s_emb", "k_emb", "v_emb"}

def gen_MainModel(model_name: os.PathLike):
    """
    从原始模型中剥离DeepEmbed相关权重，生成一个不含DE的主模型文件。
    """
    print(f"Generating MainModel without DeepEmbed from: {model_name}")
    z = torch.load(model_name if model_name.endswith(".pth") else f"{model_name}.pth", map_location="cpu", weights_only=True)
    pth = {}

    keys = list(z.keys()) # 获取原始模型的所有键
    
    # 使用 tqdm 包裹 keys 迭代器以显示进度条
    for k in tqdm(keys, desc="Processing MainModel weights"):
        # 跳过DeepEmbed相关的键
        if any(kw in k for kw in key_words): 
            continue

        pth[k] = z[k].squeeze()
        # 对特定层的权重进行转置
        if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k or 'qq.weight' in k:
            pth[k] = z[k].t()

    # LayerNorm融合到emb.weight
    pth['emb.weight'] = torch.layer_norm(z['emb.weight'], (z['emb.weight'].shape[-1],), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])

    # 这些值在推理中实际被忽略，但为了结构完整性保留
    pth['blocks.0.att.v0'] = z.get('blocks.0.att.a0', torch.zeros(1))
    pth['blocks.0.att.v1'] = z.get('blocks.0.att.a1', torch.zeros(1))
    pth['blocks.0.att.v2'] = z.get('blocks.0.att.a2', torch.zeros(1))

    output_filename = model_name.rsplit('.', maxsplit=1)[0] + "_NoDE.pth"
    torch.save(pth, output_filename)
    print(f"MainModel without DeepEmbed saved to: {output_filename}")


def get_DeepEmbed(model_name: os.PathLike):
    """
    从原始模型中提取并计算DeepEmbed权重。
    """
    print(f"Extracting and calculating DeepEmbed weights from: {model_name}")
    pth = torch.load(model_name if model_name.endswith(".pth") else f"{model_name}.pth", map_location="cpu", weights_only=True)
    
    # 动态计算模型的层数
    n_layer = 0
    for k in pth.keys():
        if k.startswith("blocks.") and ".ffn.s_emb.weight" in k:
            layer_idx = int(k.split(".")[1])
            if layer_idx + 1 > n_layer:
                n_layer = layer_idx + 1
                
    print(f"Detected model layer number: {n_layer}")

    de_pth = {}
    # 预先计算 LayerNorm 后的 embedding
    norm_emb = torch.layer_norm(pth['emb.weight'], (pth['emb.weight'].shape[-1],), weight=pth['blocks.0.ln0.weight'], bias=pth['blocks.0.ln0.bias'])
    
    for i in tqdm(range(n_layer), desc="Calculating DeepEmbed weights"):
        de_pth[f"s_emb.{i}"] = pth[f"blocks.{i}.ffn.s_emb.weight"] + norm_emb @ pth[f"blocks.{i}.ffn.s_emb_x.weight"].t()
        de_pth[f"k_emb.{i}"] = pth[f"blocks.{i}.qkv.k_emb.weight"] + norm_emb @ pth[f"blocks.{i}.qkv.k_emb_x.weight"].t()
        de_pth[f"v_emb.{i}"] = pth[f"blocks.{i}.qkv.v_emb.weight"] + norm_emb @ pth[f"blocks.{i}.qkv.v_emb_x.weight"].t()
    print("DeepEmbed weights calculation complete.")

    return de_pth


def create_data_store(output_path: os.PathLike, original_data_dict: dict):
    """
    将权重字典序列化为一个自定义的二进制格式文件（.bin）。
    该文件包含数据区、JSON索引区和文件尾部的元信息。
    """
    if os.path.exists(output_path):
        print(f"File {output_path} already exists and will be overwritten.")
    
    dtype_map = {
        torch.float32: 0, torch.float16: 1, torch.bfloat16: 2,
        torch.int64: 3, torch.int32: 4, torch.uint8: 5
    }

    final_index = {}
    
    with open(output_path, 'wb') as f:
        print("[1/3] Writing tensor data blocks...")
        for key, tensor in tqdm(original_data_dict.items(), desc="Writing tensors"):
            tensor_u8 = tensor.contiguous().view(torch.uint8)
            tensor_bytes = tensor_u8.numpy().tobytes()
            
            final_index[key] = {
                "offset": f.tell(),
                "shape": list(tensor.shape),
                "dtype": dtype_map[tensor.dtype]
            }
            f.write(tensor_bytes)

        print("[2/3] Writing JSON index...")
        index_offset = f.tell()
        index_bytes = json.dumps(final_index, separators=(',', ':')).encode('utf-8')
        f.write(index_bytes)
        index_size = len(index_bytes)

        print("[3/3] Writing footer...")
        f.write(struct.pack('<QQ', index_offset, index_size))

    print(f"Data file {output_path} built successfully!")
    print(f"Index starts at offset {index_offset}, size is {index_size} bytes.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract DeepEmbed weights from a RWKV .pth model and save them to a custom .bin file for memory mapping."
    )
    
    parser.add_argument(
        "model_path", 
        type=str, 
        help="Path to the input RWKV .pth model file."
    )
    
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="DeepEmbed.bin", 
        help="Path for the output .bin file. (default: DeepEmbed.bin)"
    )

    parser.add_argument(
        "--no-gen-main",
        action="store_true",
        help="If set, skip the generation of the main model file (_NoDE.pth)."
    )
    
    args = parser.parse_args()

    if not args.no_gen_main:
        gen_MainModel(args.model_path)
    else:
        print("Skipping the generation of the main model file as requested.")

    model_weights = get_DeepEmbed(args.model_path)
    
    create_data_store(args.output, model_weights)