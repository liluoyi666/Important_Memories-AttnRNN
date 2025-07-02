import torch
import torch.nn as nn
from models import AttnRNN


# 计算模型参数量的辅助函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 设置统一的输入和隐藏层尺寸
input_size = 2
hidden_size = 128
batch_size = 32
seq_len = 200

transformer_num_layers=2

# 创建比较的模型实例
models = {
    "Vanilla RNN": nn.RNN(input_size, hidden_size, batch_first=True),
    "LSTM": nn.LSTM(input_size, hidden_size, batch_first=True),
    "GRU": nn.GRU(input_size, hidden_size, batch_first=True),
    "AttnRNN": AttnRNN(input_size, hidden_size, batch_first=True)
    # ,
    # "transformer": nn.TransformerEncoder(
    #     nn.TransformerEncoderLayer(hidden_size, 4),
    #     transformer_num_layers)
}

# 准备测试输入数据
x = torch.randn(batch_size, seq_len, input_size)

# 计算并打印各模型参数量
print(f"模型参数量比较 (输入维度={input_size}, 隐藏维度={hidden_size}):")
print("-" * 60)
for name, model in models.items():
    # 特殊处理LSTM的初始状态
    if name == "LSTM":
        h0 = (torch.zeros(1, batch_size, hidden_size),
              torch.zeros(1, batch_size, hidden_size))
        model(x, h0)  # 前向传播以初始化参数
    else:
        model(x)  # 前向传播以初始化参数

    param_count = count_parameters(model)
    print(f"{name:<12}: {param_count:>8,} 参数")
    print(f"  参数组成: {[p.numel() for p in model.parameters()]}")

# 单独分析AttnRNN的参数量组成
print("\nAttnRNN详细参数分解:")
attn_rnn = models["AttnRNN"]
total_params = 0
for name, module in attn_rnn.named_modules():
    if isinstance(module, nn.Linear):
        mod_params = count_parameters(module)
        total_params += mod_params
        print(
            f"  {name}: {mod_params}参数 (权重: {module.weight.shape}, 偏置: {module.bias.shape if module.bias is not None else None})")
    elif isinstance(module, nn.LayerNorm):
        mod_params = count_parameters(module)
        total_params += mod_params
        print(f"  {name}: {mod_params}参数 (缩放: {module.weight.shape}, 平移: {module.bias.shape})")

print(f"AttnRNN总参数量: {total_params}参数")