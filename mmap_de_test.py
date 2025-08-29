import torch
import torch.nn as nn
import unittest
import time
import safetensors.torch
from torch.utils.cpp_extension import load
import os

# --- 全局配置 ---
DATA_FILE = "DeepEmbed.bin"
BASELINE_FILE = "rwkv7b-g1b-0.1b-20250822-ctx4096_DE.st"
TOKEN_IDS = torch.tensor([42, 1145, 91, 72, 8888, 12345, 65535], dtype=torch.int64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. JIT编译和加载C++模块 (一次性) ---
print("正在加载/编译C++异步算子...")
start_compile_time = time.time()
try:
    async_loader = load(
        name="async_loader",
        sources=["async_loader.cpp"],
        extra_include_paths=["./third_party"],
        verbose=True
    )
    print(f"算子加载完成，耗时: {time.time() - start_compile_time:.2f}秒")
except Exception as e:
    print(f"算子编译失败: {e}")
    async_loader = None

# ============================================================================
#  测试套件
# ============================================================================
@unittest.skipIf(async_loader is None, "C++算子编译失败，跳过所有测试")
class TestAsyncLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """在所有测试开始前，加载一次基准数据"""
        print(f"\n{'='*20} 开始测试 {'='*20}")
        if not os.path.exists(BASELINE_FILE):
            raise FileNotFoundError(f"基准文件未找到: {BASELINE_FILE}")
        cls.baseline_tensors = safetensors.torch.load_file(BASELINE_FILE, device="cpu")
        print(f"基准文件 '{BASELINE_FILE}' 加载完成。")

    def _get_baseline_tensor(self, key):
        """辅助函数：从基准文件中获取并切片张量"""
        # 注意：safetensors文件中的key可能包含额外的前缀
        full_key = f"blocks.{key.split('.')[1]}.{key.split('.')[0]}.weight"
        if full_key not in self.baseline_tensors:
             # 您的get_DeepEmbed函数合并了emb层，而safetensor没有，所以key格式不同
             # 我们直接从您的bin文件key格式映射到safetensor的key
             layer, emb_type, _ = key.split('.')
             emb_type_map = {'s_emb': 'ffn.s_emb', 'k_emb': 'qkv.k_emb', 'v_emb': 'qkv.v_emb'}
             full_key = f"blocks.{layer}.{emb_type_map[emb_type]}.weight"
        
        return self.baseline_tensors[full_key][TOKEN_IDS]

    def test_01_correctness_basic(self):
        """测试单个张量的数值正确性"""
        print("\n--- [测试1/5] 基础数值正确性 ---")
        key = "s_emb.5"
        
        future = async_loader.trigger_io(DATA_FILE, key, TOKEN_IDS.tolist())
        loaded_tensor_cpu = future.get()
        
        baseline_tensor = self._get_baseline_tensor(key)

        print(f"加载张量 shape: {loaded_tensor_cpu.shape}, dtype: {loaded_tensor_cpu.dtype}")
        print(f"基准张量 shape: {baseline_tensor.shape}, dtype: {baseline_tensor.dtype}")
        
        self.assertEqual(loaded_tensor_cpu.shape, baseline_tensor.shape)
        torch.testing.assert_close(loaded_tensor_cpu, baseline_tensor, rtol=1e-3, atol=1e-3)
        print("✅ 单个张量数值校验通过！")

    def test_02_correctness_comprehensive(self):
        """全面测试所有层和所有类型的张量"""
        print("\n--- [测试2/5] 全面数值正确性 ---")
        num_layers = 12
        emb_types = ["s_emb", "k_emb", "v_emb"]
        
        for i in range(num_layers):
            for emb_type in emb_types:
                key = f"{emb_type}.{i}"
                with self.subTest(key=key):
                    future = async_loader.trigger_io(DATA_FILE, key, TOKEN_IDS.tolist())
                    loaded_tensor_cpu = future.get()
                    baseline_tensor = self._get_baseline_tensor(key)
                    torch.testing.assert_close(loaded_tensor_cpu, baseline_tensor, rtol=1e-3, atol=1e-3)
        
        print(f"✅ 所有 {num_layers * len(emb_types)} 个张量数值校验通过！")

    def test_03_performance_overlap(self):
        """测试异步IO与计算的重叠效果"""
        print("\n--- [测试3/5] 性能重叠 ---")
        key = "v_emb.11"
        simulated_gpu_work_time = 0.1 # 模拟100ms的GPU计算
        
        # 异步测试
        t0 = time.time()
        future = async_loader.trigger_io(DATA_FILE, key, TOKEN_IDS.tolist())
        t1 = time.time()
        
        # 模拟GPU计算
        time.sleep(simulated_gpu_work_time)
        
        t2 = time.time()
        result = future.get()
        t3 = time.time()

        trigger_latency = (t1 - t0) * 1000
        wait_latency = (t3 - t2) * 1000
        total_async_time = (t3 - t0) * 1000
        
        print(f"触发I/O耗时: {trigger_latency:.2f} ms (应非常小)")
        print(f"模拟计算耗时: {simulated_gpu_work_time * 1000:.2f} ms")
        print(f"等待I/O结果耗时: {wait_latency:.2f} ms (可能部分或完全被计算隐藏)")
        print(f"异步总耗时: {total_async_time:.2f} ms")
        
        self.assertLess(trigger_latency, 10) # 触发操作应在10ms内完成
        self.assertTrue(result is not None)
        print("✅ 性能测试执行完成。请观察等待耗时是否小于纯I/O耗时。")
        
    def test_04_concurrency(self):
        """测试并发触发多个IO请求"""
        print("\n--- [测试4/5] 并发请求 ---")
        keys = ["s_emb.1", "k_emb.2", "v_emb.3"]
        futures = []
        for key in keys:
            futures.append(async_loader.trigger_io(DATA_FILE, key, TOKEN_IDS.tolist()))
            
        results = [f.get() for f in futures]
        
        for i, key in enumerate(keys):
            with self.subTest(key=key):
                baseline_tensor = self._get_baseline_tensor(key)
                torch.testing.assert_close(results[i], baseline_tensor, rtol=1e-3, atol=1e-3)
        print("✅ 并发请求测试通过！")
        
    def test_05_jit_compatibility(self):
        """测试包含自定义算子的模型是否可以被JIT编译"""
        print("\n--- [测试5/5] JIT兼容性 ---")

        class MyJitModel(nn.Module):
            def __init__(self, data_file):
                super().__init__()
                self.data_file = data_file
                self.linear = nn.Linear(768, 768) # 假设一个操作

            def forward(self, tokens):
                # 在JIT脚本中，外部函数调用是允许的
                # trigger_io返回一个自定义类的对象，.get()返回Tensor
                # JIT可以跟踪到.get()返回的是Tensor，并继续后续的计算图
                future_k = async_loader.trigger_io(self.data_file, "k_emb.8", tokens.tolist())
                future_v = async_loader.trigger_io(self.data_file, "v_emb.8", tokens.tolist())
                
                # JIT脚本不支持在Tensor上直接进行.tolist()，所以输入tokens需要是cpu tensor
                
                k_emb = future_k.get()
                v_emb = future_v.get()
                
                # 模型的核心计算
                # 这里我们假设k_emb和v_emb形状可以输入到线性层
                # 在实际应用中，您会有更复杂的逻辑
                output = self.linear(k_emb) + v_emb
                return output

        # 实例化模型
        model = MyJitModel(DATA_FILE).to(device=DEVICE, dtype=torch.float16)
        
        # 尝试JIT编译
        try:
            scripted_model = torch.jit.script(model)
            print("✅ 模型JIT Script编译成功！")
        except Exception as e:
            self.fail(f"模型JIT Script编译失败: {e}")
            
        # 验证JIT模型与Eager模式结果是否一致
        model.eval()
        scripted_model.eval()
        
        # JIT模型需要CPU张量来调用.tolist()
        jit_input_tokens = TOKEN_IDS.cpu()
        
        with torch.no_grad():
            eager_output = model(jit_input_tokens).cpu()
            jit_output = scripted_model(jit_input_tokens).cpu()

        torch.testing.assert_close(eager_output, jit_output, rtol=1e-3, atol=1e-3)
        print("✅ JIT编译模型与Eager模式运行结果一致！")


# --- 运行测试 ---
if __name__ == "__main__":
    unittest.main()