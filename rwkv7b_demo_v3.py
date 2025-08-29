########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
#
# This version is GPT-mode + RNN-mode, and a bit more difficult to understand
#
########################################################################################################

import copy
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, time
from typing import List
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

import torch.nn as nn
from torch.nn import functional as F
from safetensors import safe_open

if False:
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
    MyStatic = torch.jit.script
else:
    MyModule = nn.Module
    MyFunction = lambda x:x
    MyStatic = lambda x:x

########################################################################################################

# print('\nNOTE: this is very inefficient (loads all weights to VRAM, and slow KV cache). better method is to prefetch DeepEmbed from RAM/SSD\n')

args = types.SimpleNamespace()

# model download: https://huggingface.co/BlinkDL/rwkv7-g1 // please compare with rwkv_v7_demo_fast.py

args.MODEL_NAME = "rwkv7b-g1b-0.1b-20250822-ctx4096"

args.n_layer = 12
args.n_embd = 768
args.vocab_size = 65536
args.head_size = 64

prompt = "Assistant: <think>地球是人类唯一的家园"
NUM_TRIALS = 1
CTX_LEN = 256
TEMPERATURE = 1.0
TOP_P = 0.0

########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

DTYPE = torch.half

from torch.utils.cpp_extension import load
HEAD_SIZE = args.head_size

load(name="wkv7s", sources=["cuda/wkv7s_op.cpp", f"cuda/wkv7s.cu"], is_python_module=False,
                    verbose=False, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
class WKV_7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
            assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
            y = torch.empty((T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.wkv7s.forward(1, T, C, H, state, r, w, k, v, a, b, y)
            return y
def RWKV7_OP(state, r, w, k, v, a, b):
    return WKV_7.apply(state, r, w, k, v, a, b)

########################################################################################################

load(name="async_loader", sources=["async_loader.cpp"], extra_include_paths=["./third_party"], verbose=False)

class DeepEmbeding():
    def __init__(self, data_file:str):
        self.data_file = data_file

    def get_de(self, key_name:str, tokens:List[int], device='cpu', dtype=DTYPE) -> torch.Tensor:
        return torch.ops.custom.perform_io_task_sync(self.data_file, key_name, tokens).to(device=device, dtype=dtype)

class RWKV_x070(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        self.eval()
        
        self.z = torch.load(args.MODEL_NAME + '.pth', map_location='cuda', weights_only=True)
        self.de = DeepEmbeding("DeepEmbed.bin")
        
        z = self.z
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape

        keys = list(z.keys())
        for k in keys:
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k or 'qq.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE)
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
        assert self.head_size == args.head_size

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])

        for i in range(self.n_layer): # !!! merge emb residual !!!
            z[f'blocks.{i}.ffn.s_emb.weight'] = z[f'blocks.{i}.ffn.s_emb.weight'] + z['emb.weight'] @ z[f'blocks.{i}.ffn.s_emb_x.weight'].t()
            z[f'blocks.{i}.qkv.k_emb.weight'] = z[f'blocks.{i}.qkv.k_emb.weight'] + z['emb.weight'] @ z[f'blocks.{i}.qkv.k_emb_x.weight'].t()
            z[f'blocks.{i}.qkv.v_emb.weight'] = z[f'blocks.{i}.qkv.v_emb.weight'] + z['emb.weight'] @ z[f'blocks.{i}.qkv.v_emb_x.weight'].t()

        z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
        z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
        z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored

    def forward(self, idx, state, full_output=False):
        idx_list = [idx] if type(idx) == int else idx
        
        if state == None:            
            state = [None for _ in range(args.n_layer * 3 + 37)] # with KV cache etc.
            for i in range(args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                state[i*3+0] = torch.zeros(args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")
                state[i*3+1] = torch.zeros((args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device="cuda")
                state[i*3+2] = torch.zeros(args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")
            state[args.n_layer*3+0] = [] # token idx cache
            for i in range(1,1+24): # kv cache = 12*2*32 numbers per token
                state[args.n_layer*3+i] = torch.empty((0,32), dtype=DTYPE, requires_grad=False, device="cuda")
            for i in range(1+24,1+36): # token-shift cache for Q in DEA
                state[args.n_layer*3+i] = torch.zeros(256, dtype=DTYPE, requires_grad=False, device="cuda")

        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                return self.forward_one(idx[0], state)

        else:
            return self.forward_one(idx, state)

    @MyFunction
    def forward_one(self, idx:int, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]
            state[self.n_layer*3] += [idx]
            ctx = state[self.n_layer*3]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'{bbb}att.'
                ffn = f'{bbb}ffn.'
                qkv = f'{bbb}qkv.'

                qkv, state[self.n_layer*3+1+24+i], state[self.n_layer*3+1+i*2], state[self.n_layer*3+1+i*2+1] = RWKV_v7s_DEA(
                    x, z[qkv+'qq.weight'], state[self.n_layer*3+1+24+i], 
                    z[qkv+'k1'], z[qkv+'k2'], state[self.n_layer*3+1+i*2], self.de.get_de(f"k_emb.{i}", ctx, device=x.device),
                    z[qkv+'v1'], z[qkv+'v2'], state[self.n_layer*3+1+i*2+1], self.de.get_de(f"v_emb.{i}", ctx, device=x.device),
                    z[qkv+'x_q'], z[qkv+'x_k'], z[qkv+'x_v'], z[qkv+'lnq.weight'], z[qkv+'lnq.bias'],
                    z[qkv+'lnk.weight'], z[qkv+'lnk.bias'], z[qkv+'lnv.weight'], z[qkv+'lnv.bias'])

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx + qkv

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = RWKV_x070_CMix_one(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'], self.de.get_de(f"s_emb.{i}", [idx], device=x.device),
                     z[ffn+'s1'], z[ffn+'s2'], z[ffn+'s0'])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state

    @MyFunction
    def forward_seq(self, idx:List[int], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]
            state[self.n_layer*3] += idx
            ctx = state[self.n_layer*3]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'{bbb}att.'
                ffn = f'{bbb}ffn.'
                qkv = f'{bbb}qkv.'

                qkv, state[self.n_layer*3+1+24+i], state[self.n_layer*3+1+i*2], state[self.n_layer*3+1+i*2+1] = RWKV_v7s_DEA(
                    x, z[qkv+'qq.weight'], state[self.n_layer*3+1+24+i], 
                    z[qkv+'k1'], z[qkv+'k2'], state[self.n_layer*3+1+i*2], self.de.get_de(f"k_emb.{i}", ctx, device=x.device),
                    z[qkv+'v1'], z[qkv+'v2'], state[self.n_layer*3+1+i*2+1], self.de.get_de(f"v_emb.{i}", ctx, device=x.device),
                    z[qkv+'x_q'], z[qkv+'x_k'], z[qkv+'x_v'], z[qkv+'lnq.weight'], z[qkv+'lnq.bias'],
                    z[qkv+'lnk.weight'], z[qkv+'lnk.bias'], z[qkv+'lnv.weight'], z[qkv+'lnv.bias'])

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx + qkv

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = RWKV_x070_CMix_seq(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'], self.de.get_de(f"s_emb.{i}", idx, device=x.device),
                     z[ffn+'s1'], z[ffn+'s2'], z[ffn+'s0'])
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state

########################################################################################################

@MyStatic
def RWKV_v7s_DEA(x, Q_, q_prev, k1, k2, k_c, k_e, v1, v2, v_c, v_e, x_q, x_k, x_v, lnq_w, lnq_b, lnk_w, lnk_b, lnv_w, lnv_b):
    is_seq_mode = x.ndim == 2
    
    q = x @ Q_
    k_proj = x @ k1
    v_proj = x @ v1

    k_c = torch.cat((k_c, k_proj.view(-1, k_proj.shape[-1])), dim=0)
    k = (k_c @ k2) * k_e

    v_c = torch.cat((v_c, v_proj.view(-1, k_proj.shape[-1])), dim=0)
    v = torch.tanh(v_c @ v2) * v_e
    

    qq = torch.cat((q_prev.unsqueeze(0), q[:-1, :])) if is_seq_mode else q_prev
    q = q + (qq - q) * x_q

    k = k + (F.pad(k, (0, 0, 1, -1)) - k) * x_k
    v = v + (F.pad(v, (0, 0, 1, -1)) - v) * x_v

    q = F.layer_norm(q, (256,), weight=lnq_w, bias=lnq_b)
    k = F.layer_norm(k, (256,), weight=lnk_w, bias=lnk_b)
    v = F.layer_norm(v, (x.shape[-1],), weight=lnv_w, bias=lnv_b)

    scores = 64 * torch.tanh((q @ k.mT) * (1.0 / 1024.0))
    if is_seq_mode:
        T = x.shape[-2]
        ctx_len = k.shape[0]  # 从 k 的实际长度推断总上下文长度
        mask = ~torch.tril(torch.ones(ctx_len, ctx_len, dtype=torch.bool, device=x.device))[-T:, :]
        scores = scores.masked_fill(mask, float('-inf'))
    
    qkv = scores.softmax(dim=-1) @ v
    
    return qkv, q[-1, :] if is_seq_mode else q, k_c, v_c

########################################################################################################

@MyStatic
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    xx = (state.to(dtype=x.dtype) @ r.view(H,N,1))

    xx = torch.nn.functional.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return (xx * g) @ O_, x, state, v_first

@MyStatic
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    T = x.shape[0]
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    # ####### cuda-free method 
    # w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
    # for t in range(T):
    #     r_, w_, k_, v_, kk_, a_ = r[t], w[t], k[t], v[t], kk[t], a[t]
    #     vk = v_.view(H,N,1) @ k_.view(H,1,N)
    #     ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
    #     state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
    #     xx[t] = (state.to(dtype=x.dtype) @ r_.view(H,N,1)).view(H*N)

    ####### cuda-kernel method 
    w = -torch.nn.functional.softplus(-(w0 + w)) - 0.5
    xx = RWKV7_OP(state, r, w, k, v, -kk, kk*a)

    xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return (xx * g) @ O_, x[-1,:], state, v_first

########################################################################################################

@MyStatic
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_, semb_, s1_, s2_, s0_):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    ss = (x @ s1_) @ semb_.view(32,32)
    k = k * ((ss @ s2_) + s0_)
    return k @ V_, x

@MyStatic
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_, semb_, s1_, s2_, s0_):
    T,C = x.shape
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2    
    ss = (x @ s1_).view(T,1,32) @ semb_.view(T,32,32)
    k = k * ((ss.view(T,32) @ s2_) + s0_)
    return k @ V_, x[-1,:]

########################################################################################################
#
# The testing code
#
########################################################################################################

@MyStatic
def sample_logits(logits, temperature:float=1.0, top_p:float=1.0, top_k:int=0):
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    
    if top_k > 0:
        probs[sorted_ids[top_k:]] = 0

    if top_p < 1:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff = sorted_probs[cutoff_index]
        probs[probs < cutoff] = 0

        if top_p > 0:
            idx = torch.where(probs == cutoff)[0]
            if len(idx) > 0:
                probs[idx] = cutoff + (top_p - torch.sum(probs).item()) / len(idx)
                # assert abs(torch.sum(probs).item() - top_p) < 1e-6
    
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    return torch.multinomial(probs, num_samples=1).item()

########################################################################################################
# RWKV Tokenizer (slow version)
########################################################################################################

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

tokenizer = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")

########################################################################################################

# print(tokenizer.decode([12509]))

time_start = time.time()

print(f'\nUsing CUDA {str(DTYPE).replace("torch.","")}. Loading {args.MODEL_NAME} ...')
model = RWKV_x070(args)

print("Models OK!")
# input("Stop point 1")

_, state = model.forward(tokenizer.encode(prompt[:-5]), None)
out, state = model.forward(tokenizer.encode(prompt[-5:]), state)

print("Prefill OK!")
# input("Stop point 2")

print(f'\n{prompt}', end='')

occurrence = {}
out_tokens = []
out_last = 0

for i in range(CTX_LEN):
    for n in occurrence:
        out[n] -= 0.5 + occurrence[n] * 0.5 # repetition penalty
        out[0] -= 1e10  # disable END_OF_TEXT

    token = sample_logits(out, TEMPERATURE, TOP_P)
    out_tokens += [token]

    out, state = model.forward(token, state)

    for n in occurrence:
        occurrence[n] *= 0.996
    occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

    try:
        tmp = tokenizer.decode(out_tokens[out_last:])
        if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):  # only print & update out_last when it's a valid utf-8 string and not ending with \n
            print(tmp, end="", flush=True)
            out_last = i + 1

        if "\n\n" in tmp:
            print(tmp, end="", flush=True)
            break
    except:
        pass

print(f"\nDecode OK!\n总耗时：{(time.time() - time_start):.2f}")
# input("Stop point 3")

########################################################################################################

# print(f'\nUsing CUDA {str(DTYPE).replace("torch.","")}. Loading {args.MODEL_NAME} ...')
# model = RWKV_x070(args)

# import json, math
# with open(f"lambada_test.jsonl", "r", encoding="utf-8") as f:
#     todo = [json.loads(line) for line in f]
#     todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

# print('\nCheck LAMBADA...')
# xsum = 0
# xcnt = 0
# xacc = 0
# for d in todo:
#     src = [0] + tokenizer.encode(d[0])
#     dst = tokenizer.encode(d[1])

#     logits = 0
#     correct = True
    
#     out, _ = model.forward(src+dst, None, full_output=True)

#     for i in range(len(dst)):
#         ooo = out[len(src)-1+i].float()
#         probs = F.softmax(ooo, dim=-1)
#         logits += math.log(probs[dst[i]])
#         if torch.argmax(probs).item() != dst[i]:
#             correct = False

#     xcnt += 1
#     xsum += logits
#     xacc += 1 if correct else 0
#     if xcnt % 100 == 0 or xcnt == len(todo):
#         print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 2))