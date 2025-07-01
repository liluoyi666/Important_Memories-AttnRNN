import torch
import torch.nn as nn
import torch.nn.functional as F
from models import AttnRNN

import torch
import torch.nn as nn
import time


class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入门权重
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        # 隐藏门权重
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        # 偏置
        self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier均匀初始化
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, x, hx):
        # 输入变换
        gates = F.linear(x, self.weight_ih, self.bias_ih) + \
                F.linear(hx, self.weight_hh, self.bias_hh)

        # 分割门值
        reset_gate, update_gate, new_gate = gates.chunk(3, 1)

        # 激活函数
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        new_gate = torch.tanh(reset_gate * (new_gate))

        # 计算新隐藏状态
        hy = (1 - update_gate) * hx + update_gate * new_gate
        return hy


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = CustomGRUCell(input_size, hidden_size)

    def forward(self, x, h0=None):
        if not self.batch_first:
            x = x.permute(1, 0, 2)

        batch_size, seq_len, _ = x.shape

        # 初始化隐藏状态
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size,
                            device=x.device, dtype=x.dtype)
        else:
            h = h0

        # 存储所有时间步的输出
        outputs = []
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
            outputs.append(h)

        output = torch.stack(outputs, dim=1)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output, h


# 时间测试函数
def time_test(model, input_size, seq_len, batch_size, hidden_size, device, num_runs=100):
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, input_size).to(device)

    # 预热GPU
    for _ in range(10):
        model(x)

    # 精确计时
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_runs):
        model(x)

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    return elapsed / num_runs


# 测试配置
input_size = 128
hidden_size = 256
seq_len = 100
batch_size = 64
device = torch.device('cuda')
num_runs = 100

# 创建模型
attn_rnn = AttnRNN(input_size, hidden_size).to(device)
custom_gru = CustomGRU(input_size, hidden_size).to(device)

# 测试时间
attn_time = time_test(attn_rnn, input_size, seq_len, batch_size, hidden_size, device, num_runs)
gru_time = time_test(custom_gru, input_size, seq_len, batch_size, hidden_size, device, num_runs)

print(f"AttnRNN平均时间: {attn_time:.6f}秒")
print(f"CustomGRU平均时间: {gru_time:.6f}秒")
print(f"时间比 (AttnRNN/GRU): {attn_time / gru_time:.2f}")