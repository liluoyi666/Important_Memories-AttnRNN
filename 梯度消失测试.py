import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# 设置全局随机种子
def set_seed(seed=54):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from models import AttnRNN


# 梯度分析器类
class GradientAnalyzer:
    def __init__(self, hidden_size=64, device='cpu'):
        self.hidden_size = hidden_size
        self.device = device
        self.models = {
            "AttnRNN": AttnRNN(input_size=hidden_size, hidden_size=hidden_size),
            "RNN": nn.RNN(input_size=hidden_size, hidden_size=hidden_size,
                          nonlinearity='tanh', batch_first=True),
            "LSTM": nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            batch_first=True),
            "GRU": nn.GRU(input_size=hidden_size, hidden_size=hidden_size,
                          batch_first=True)
        }
        # 将模型移到设备
        for name, model in self.models.items():
            model.to(device)

    def analyze_gradients(self, seq_lengths, num_trials=5):
        results = {name: [] for name in self.models}

        for length in tqdm(seq_lengths, desc="Testing sequence lengths"):
            # 初始化每个模型在当前序列长度的梯度范数列表
            grad_norms = {name: [] for name in self.models}

            for _ in range(num_trials):
                # 生成输入数据和目标
                x = torch.ones(1, length, self.hidden_size, device=self.device)  # (batch, seq, features)
                target = torch.ones(1, self.hidden_size, device=self.device)  # 目标张量 (1, hidden_size)

                for name, model in self.models.items():
                    # 重置模型参数和梯度
                    model.zero_grad()

                    # 初始化隐藏状态（可学习）
                    if name == "LSTM":
                        h0 = torch.zeros(1, 1, self.hidden_size, device=self.device, requires_grad=True)
                        c0 = torch.zeros(1, 1, self.hidden_size, device=self.device, requires_grad=True)
                        hidden = (h0, c0)
                    elif name == "AttnRNN":
                        h0 = torch.zeros(1, self.hidden_size, device=self.device, requires_grad=True)
                    else:  # RNN和GRU
                        h0 = torch.zeros(1, 1, self.hidden_size, device=self.device, requires_grad=True)
                        hidden = h0

                    # 前向传播
                    if name == "AttnRNN":
                        output, h_final = model(x, h0=h0)
                        # 确保形状匹配 (1, hidden_size)
                        h_final = h_final.unsqueeze(0) if len(h_final.shape) == 1 else h_final
                    elif name == "LSTM":
                        output, (h_final, c_final) = model(x, hidden)
                        h_final = h_final.squeeze(0)  # 从 (1,1,hidden_size) 转为 (1, hidden_size)
                    else:  # RNN和GRU
                        output, h_final = model(x, hidden)
                        h_final = h_final.squeeze(0)  # 从 (1,1,hidden_size) 转为 (1, hidden_size)

                    # 计算损失（确保形状匹配）
                    loss = F.mse_loss(h_final, target)

                    # 反向传播
                    loss.backward()

                    # 计算初始隐藏状态的梯度范数
                    if name == "LSTM":
                        grad_norm = h0.grad.norm().item()
                    else:
                        grad_norm = h0.grad.norm().item()
                    grad_norms[name].append(grad_norm)

            # 计算当前序列长度的平均梯度范数
            for name in self.models:
                avg_grad = np.mean(grad_norms[name])
                results[name].append(avg_grad)

        return results

    def plot_results(self, results, seq_lengths, filename="gradient_comparison.png"):
        plt.figure(figsize=(12, 8))

        for name, norms in results.items():
            # 处理零值避免对数无穷大
            safe_norms = [max(norm, 1e-16) for norm in norms]
            plt.plot(seq_lengths, safe_norms, 'o-', label=name, linewidth=2.5)

        plt.xlabel('Sequence Length', fontsize=14)
        plt.ylabel('Gradient Norm (log scale)', fontsize=14)
        plt.title('Gradient Vanishing Comparison', fontsize=16)
        plt.yscale('log')
        plt.grid(True, which="both", linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        # plt.savefig(filename, dpi=300)
        plt.show()


# 配置参数
if __name__ == "__main__":
    # 设置全局随机种子
    set_seed(54)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 设置序列长度范围 (从短到长)
    seq_lengths = [ 16,32,64,96,128,160,192,224,256,288,320,352,384,416,448,480,512,1024]

    # 创建分析器
    analyzer = GradientAnalyzer(hidden_size=64, device=device)

    # 分析梯度
    results = analyzer.analyze_gradients(seq_lengths, num_trials=3)

    # 绘制并保存结果
    analyzer.plot_results(results, seq_lengths, "rnn_gradient_comparison.png")

    # 打印结果表格
    print("\n梯度范数结果:")
    print("Length\t" + "\t".join(analyzer.models.keys()))
    for i, length in enumerate(seq_lengths):
        row = [f"{results[name][i]:.4e}" for name in analyzer.models]
        print(f"{length}\t" + "\t".join(row))