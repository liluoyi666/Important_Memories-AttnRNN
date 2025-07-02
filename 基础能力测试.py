import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 仅投影Q值
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # KV不使用投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, context):
        batch_size = query.size(0)

        # 仅投影Q值
        q = self.q_proj(query)  # (batch, q_len, embed_dim)

        # K和V直接使用原始context
        k = context  # (batch, ctx_len, embed_dim)
        v = context  # (batch, ctx_len, embed_dim)

        # 重塑为多头格式
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embed_dim)

        return self.out_proj(attn_output) # 只取query对应部分


class AttnRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # 投影层
        self.proj_layer = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )

        # 注意力机制
        self.attn = EfficientAttention(
            embed_dim=hidden_size,
            num_heads=num_heads
        )

        # 门控机制
        self.update_gate = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, h_prev, x):
        x_proj = self.proj_layer(x)

        # 准备注意力输入
        h_exp = h_prev.unsqueeze(1)
        x_exp = x_proj.unsqueeze(1)
        context = torch.cat([h_exp, x_exp, h_exp + x_exp, h_exp * x_exp], dim=1)

        # 注意力计算
        attn_out = self.attn(query=h_exp, context=context)
        attn_out = attn_out.squeeze(1)  # [batch, hidden]
        attn_out = F.gelu(attn_out)  # 添加 GELU 激活

        # 门控融合
        gate_input = torch.cat([h_prev, attn_out], dim=-1)
        update_gate = self.update_gate(gate_input)

        # 门控更新
        h_candidate = self.layer_norm(attn_out + h_prev)  # 原残差路径
        h_new = update_gate * h_candidate + (1 - update_gate) * h_prev

        return h_new

class AttnRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True, init_value=0.001):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.init_value = init_value
        self.cell = AttnRNNCell(input_size, hidden_size)

    def forward(self, x, h0=None):
        if not self.batch_first:
            x = x.permute(1, 0, 2)

        batch_size, seq_len, _ = x.shape

        # 初始化隐藏状态
        if h0 is not None:
            h = h0
        else:
            h = torch.full(
                (batch_size, self.hidden_size),
                self.init_value,
                device=x.device,
                dtype=x.dtype
            )

        outputs = []
        for t in range(seq_len):
            h = self.cell(h, x[:, t, :])
            outputs.append(h)

        output = torch.stack(outputs, dim=1)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output, h

# 生成连续模式检测任务数据
def generate_data(batch_size, seq_length, num_classes, noise_level=0.0):
    """
    生成连续模式检测任务数据:
    - 序列包含整数（0到num_classes-1）
    - 每个序列包含一个连续5个相同值的模式（答案）
    - 答案只在第五个位置及之后才被识别
    - 目标: 在序列结束时输出答案类别
    """
    # 创建空序列
    sequences = np.random.randint(0, num_classes, (batch_size, seq_length))
    targets = np.zeros(batch_size, dtype=int)

    for i in range(batch_size):
        # 随机选择答案位置（确保有足够的空间放置5个连续值）
        start_pos = np.random.randint(0, seq_length - 4)

        # 随机选择答案值
        answer = np.random.randint(0, num_classes)
        targets[i] = answer

        # 创建连续5个相同值
        sequences[i, start_pos:start_pos + 5] = answer

        # 添加噪声（随机修改部分值）
        noise_positions = np.random.choice(seq_length, int(seq_length * noise_level), replace=False)
        for pos in noise_positions:
            # 避免修改答案模式
            if pos < start_pos or pos >= start_pos + 5:
                # 随机选择一个不同于当前值的新值
                new_val = np.random.choice([x for x in range(num_classes) if x != sequences[i, pos]])
                sequences[i, pos] = new_val

    # 转换为one-hot编码
    input_size = num_classes
    sequences_onehot = np.zeros((batch_size, seq_length, input_size))
    for i in range(batch_size):
        sequences_onehot[i] = np.eye(input_size)[sequences[i]]

    return (
        torch.tensor(sequences_onehot, dtype=torch.float32).to(Config.device),
        torch.tensor(targets, dtype=torch.long).to(Config.device)
    )


# 模型定义（添加GRU）
class BaseRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, Config.num_classes)

    def forward(self, x):
        output, _ = self.rnn(x)
        # 只取序列最后一个时间步的输出
        return self.fc(output[:, -1, :])


class BaseGRU(nn.Module):  # 新增GRU模型
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, Config.num_classes)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])


class BaseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, Config.num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

class AttnRNNWrapper(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = AttnRNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, Config.num_classes)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])


# 训练和评估函数（修复Config.input_size错误）
def train_and_evaluate(model, seq_length, noise_level=0.0):
    model.to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    model.train()
    for epoch in range(Config.num_epochs):
        total_loss = 0
        for _ in range(100):  # 每epoch 100个batch
            # 修复这里：使用Config.num_classes而不是Config.input_size
            inputs, targets = generate_data(
                Config.batch_size, seq_length,
                Config.num_classes, noise_level  # 只传递三个参数
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # 添加梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        # 打印训练进度
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{Config.num_epochs}, Loss: {total_loss / 100:.4f}")

    # 评估
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for _ in range(20):  # 20个测试batch
            inputs, targets = generate_data(
                Config.batch_size, seq_length,
                Config.num_classes, noise_level  # 只传递三个参数
            )
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    return acc


# 添加问题(Adding Problem)数据生成
def generate_adding_data(batch_size, seq_length):
    """
    生成Adding Problem数据:
    - 输入序列: (value, indicator) pairs
    - 随机选择两个位置indicator=1，其余为0
    - 目标: 两个标记位置value的和
    """
    values = np.random.uniform(0, 1, (batch_size, seq_length, 1))
    indicators = np.zeros((batch_size, seq_length, 1))

    # 随机选择两个位置设置indicator=1
    for i in range(batch_size):
        positions = np.random.choice(seq_length, size=2, replace=False)
        indicators[i, positions] = 1

    inputs = np.concatenate([values, indicators], axis=-1)
    targets = np.sum(values * indicators, axis=1)  # 两个标记值的和

    return (
        torch.tensor(inputs, dtype=torch.float32).to(Config.device),
        torch.tensor(targets, dtype=torch.float32).to(Config.device)
    )


# 复制记忆任务(Copy Memory Task)数据生成
def generate_copy_data(batch_size, seq_length, num_classes=8):
    """
    生成Copy Memory Task数据:
    - 输入序列: [data] + [delimiter] + [silence]
    - 目标: 复制数据段内容到输出序列的相应位置
    """
    data_length = seq_length // 3
    delimiter_length = 1
    silence_length = seq_length - data_length - delimiter_length

    # 创建序列
    sequences = np.zeros((batch_size, seq_length), dtype=int)
    targets = np.zeros((batch_size, seq_length), dtype=int)

    for i in range(batch_size):
        # 生成数据段 (类别1-7)
        data = np.random.randint(1, num_classes, data_length)
        sequences[i, :data_length] = data

        # 分隔符 (类别0)
        sequences[i, data_length] = 0

        # 静默段 (类别8)
        sequences[i, data_length + 1:] = num_classes - 1

        # 目标序列: 在静默段复制数据
        targets[i, -data_length:] = data

    # 转换为one-hot编码
    input_size = num_classes
    sequences_onehot = np.zeros((batch_size, seq_length, input_size))
    targets_onehot = np.zeros((batch_size, seq_length, input_size))

    for i in range(batch_size):
        sequences_onehot[i] = np.eye(input_size)[sequences[i]]
        targets_onehot[i] = np.eye(input_size)[targets[i]]

    return (
        torch.tensor(sequences_onehot, dtype=torch.float32).to(Config.device),
        torch.tensor(targets_onehot, dtype=torch.float32).to(Config.device)
    )


# 修改模型以处理序列输出
class SequenceModelWrapper(nn.Module):
    """包装器使模型适应序列到序列任务"""

    def __init__(self, base_model, output_size):
        super().__init__()
        self.base_model = base_model
        self.output_layer = nn.Linear(Config.hidden_size, output_size)

    def forward(self, x):
        # 对于RNN/GRU/LSTM
        if hasattr(self.base_model, 'rnn') or hasattr(self.base_model, 'gru') or hasattr(self.base_model, 'lstm'):
            output, _ = self.base_model.rnn(x) if hasattr(self.base_model, 'rnn') else \
                self.base_model.gru(x) if hasattr(self.base_model, 'gru') else \
                    self.base_model.lstm(x)
            return self.output_layer(output)

        # 对于AttnRNN
        output, _ = self.base_model.rnn(x)
        return self.output_layer(output)


# 修改训练函数以支持不同任务
def train_and_evaluate(model, task='pattern', seq_length=128, noise_level=0.0):
    model.to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    # 根据任务选择损失函数
    if task == 'adding':
        criterion = nn.MSELoss()
    else:  # pattern或copy任务
        criterion = nn.CrossEntropyLoss() if task == 'pattern' else nn.CrossEntropyLoss(ignore_index=0)

    # 训练循环
    model.train()
    for epoch in range(Config.num_epochs):
        total_loss = 0
        for _ in range(100):  # 每epoch 100个batch
            # 生成不同任务的数据
            if task == 'pattern':
                inputs, targets = generate_data(
                    Config.batch_size, seq_length, Config.num_classes, noise_level)
            elif task == 'adding':
                inputs, targets = generate_adding_data(Config.batch_size, seq_length)
            else:  # copy任务
                inputs, targets = generate_copy_data(Config.batch_size, seq_length, Config.num_classes)

            optimizer.zero_grad()
            outputs = model(inputs)

            # 计算不同任务的损失
            if task == 'adding':
                loss = criterion(outputs.squeeze(), targets.squeeze())
            elif task == 'pattern':
                loss = criterion(outputs, targets)
            else:  # copy任务
                # 只计算静默段的损失
                data_length = seq_length // 3
                valid_positions = slice(-data_length, None)
                loss = criterion(outputs[:, valid_positions, :].reshape(-1, Config.num_classes),
                                 targets[:, valid_positions].argmax(-1).reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # 打印训练进度
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{Config.num_epochs}, Loss: {total_loss / 100:.4f}")

    # 评估
    model.eval()
    with torch.no_grad():
        test_inputs, test_targets = generate_adding_data(100, seq_length) if task == 'adding' else \
            generate_copy_data(100, seq_length, Config.num_classes) if task == 'copy' else \
                generate_data(100, seq_length, Config.num_classes, noise_level)

        test_outputs = model(test_inputs)

        if task == 'adding':
            # 计算MAE
            mae = torch.abs(test_outputs.squeeze() - test_targets.squeeze()).mean().item()
            print(f"Adding Task MAE: {mae:.4f}")
            return mae

        elif task == 'pattern':
            # 计算准确率
            preds = torch.argmax(test_outputs, dim=1)
            targets_labels = test_targets
            acc = accuracy_score(targets_labels.cpu().numpy(), preds.cpu().numpy())
            print(f"Pattern Detection Accuracy: {acc:.4f}")
            return acc

        else:  # copy任务
            # 计算复制准确率
            data_length = seq_length // 3
            valid_positions = slice(-data_length, None)
            preds = torch.argmax(test_outputs[:, valid_positions, :], dim=-1)
            targets_labels = test_targets[:, valid_positions].argmax(-1)
            acc = (preds == targets_labels).float().mean().item()
            print(f"Copy Task Accuracy: {acc:.4f}")
            return acc


# 主实验函数
def run_memory_experiments():
    results = {
        "pattern": {"lengths": {}, "noise": {}},
        "adding": {"lengths": {}},
        "copy": {"lengths": {}}
    }

    # 定义模型集合
    models = {
        "RNN": BaseRNN(Config.num_classes, Config.hidden_size),
        "GRU": BaseGRU(Config.num_classes, Config.hidden_size),
        "LSTM": BaseLSTM(Config.num_classes, Config.hidden_size),
        "AttnRNN": AttnRNNWrapper(Config.num_classes, Config.hidden_size)
    }

    # 1. 模式检测任务 (原始任务)
    print("\n" + "=" * 50)
    print("Running Pattern Detection Experiments")
    print("=" * 50)

    # 序列长度影响
    print("\nTesting sequence length impact on pattern detection...")
    for seq_len in Config.pattern_lengths:
        print(f"\nSequence length: {seq_len}")
        seq_results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            acc = train_and_evaluate(model, task='pattern', seq_length=seq_len)
            seq_results[name] = acc
            print(f"{name}: Accuracy = {acc:.4f}")
        results["pattern"]["lengths"][seq_len] = seq_results

    # 噪声影响
    print("\nTesting noise impact on pattern detection...")
    for noise in Config.noise_levels:
        print(f"\nNoise level: {noise}")
        noise_results = {}
        for name in models.keys():
            # 重新初始化模型
            if name == "RNN":
                model = BaseRNN(Config.num_classes, Config.hidden_size)
            elif name == "LSTM":
                model = BaseLSTM(Config.num_classes, Config.hidden_size)
            elif name == "GRU":
                model = BaseGRU(Config.num_classes, Config.hidden_size)
            else:
                model = AttnRNNWrapper(Config.num_classes, Config.hidden_size)

            print(f"Training {name}...")
            acc = train_and_evaluate(model, task='pattern', seq_length=128, noise_level=noise)
            noise_results[name] = acc
            print(f"{name}: Accuracy = {acc:.4f}")
        results["pattern"]["noise"][noise] = noise_results

    # 2. Adding Problem 任务
    print("\n" + "=" * 50)
    print("Running Adding Problem Experiments")
    print("=" * 50)

    # 修改模型为序列输出
    seq_models = {}
    for name, model in models.items():
        # 对于添加问题，我们只需要标量输出
        if name == "RNN":
            base = BaseRNN(2, Config.hidden_size)
            base.fc = nn.Linear(Config.hidden_size, 1)
        elif name == "LSTM":
            base = BaseLSTM(2, Config.hidden_size)
            base.fc = nn.Linear(Config.hidden_size, 1)
        elif name == "GRU":
            base = BaseGRU(2, Config.hidden_size)
            base.fc = nn.Linear(Config.hidden_size, 1)
        else:  # AttnRNN
            base = AttnRNNWrapper(2, Config.hidden_size)
            base.fc = nn.Linear(Config.hidden_size, 1)
        seq_models[name] = base

    # 序列长度影响
    print("\nTesting sequence length impact on Adding Problem...")
    for seq_len in Config.adding_lengths:
        print(f"\nSequence length: {seq_len}")
        seq_results = {}
        for name, model in seq_models.items():
            print(f"Training {name}...")
            mae = train_and_evaluate(model, task='adding', seq_length=seq_len)
            seq_results[name] = mae
            print(f"{name}: MAE = {mae:.4f}")
        results["adding"]["lengths"][seq_len] = seq_results

    # 3. Copy Memory 任务
    print("\n" + "=" * 50)
    print("Running Copy Memory Task Experiments")
    print("=" * 50)

    # 修改模型为序列输出
    copy_models = {}
    for name, model in models.items():
        if name == "RNN":
            copy_models[name] = SequenceModelWrapper(BaseRNN(Config.num_classes, Config.hidden_size),
                                                     Config.num_classes)
        elif name == "LSTM":
            copy_models[name] = SequenceModelWrapper(BaseLSTM(Config.num_classes, Config.hidden_size),
                                                     Config.num_classes)
        elif name == "GRU":
            copy_models[name] = SequenceModelWrapper(BaseGRU(Config.num_classes, Config.hidden_size),
                                                     Config.num_classes)
        else:  # AttnRNN
            copy_models[name] = SequenceModelWrapper(AttnRNNWrapper(Config.num_classes, Config.hidden_size),
                                                     Config.num_classes)

    # 序列长度影响
    print("\nTesting sequence length impact on Copy Memory Task...")
    for seq_len in Config.copy_lengths:
        print(f"\nSequence length: {seq_len}")
        seq_results = {}
        for name, model in copy_models.items():
            print(f"Training {name}...")
            acc = train_and_evaluate(model, task='copy', seq_length=seq_len)
            seq_results[name] = acc
            print(f"{name}: Accuracy = {acc:.4f}")
        results["copy"]["lengths"][seq_len] = seq_results

    return results


# 结果可视化
def visualize_results(results):
    # 创建三个子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. 模式检测任务 - 序列长度影响
    pattern_lengths = sorted(results["pattern"]["lengths"].keys())
    for model in results["pattern"]["lengths"][pattern_lengths[0]].keys():
        accs = [results["pattern"]["lengths"][l][model] for l in pattern_lengths]
        axes[0].plot(pattern_lengths, accs, marker='o', label=model)
    axes[0].set_title("Pattern Detection (Accuracy vs Seq Length)")
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # 2. Adding Problem - 序列长度影响
    adding_lengths = sorted(results["adding"]["lengths"].keys())
    for model in results["adding"]["lengths"][adding_lengths[0]].keys():
        maes = [results["adding"]["lengths"][l][model] for l in adding_lengths]
        axes[1].plot(adding_lengths, maes, marker='o', label=model)
    axes[1].set_title("Adding Problem (MAE vs Seq Length)")
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("Mean Absolute Error")
    axes[1].legend()
    axes[1].grid(True)

    # 3. Copy Memory Task - 序列长度影响
    copy_lengths = sorted(results["copy"]["lengths"].keys())
    for model in results["copy"]["lengths"][copy_lengths[0]].keys():
        accs = [results["copy"]["lengths"][l][model] for l in copy_lengths]
        axes[2].plot(copy_lengths, accs, marker='o', label=model)
    axes[2].set_title("Copy Memory Task (Accuracy vs Seq Length)")
    axes[2].set_xlabel("Sequence Length")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("memory_experiments_results.png")
    plt.show()


# 更新实验配置
class Config:
    num_classes = 8  # 整数类别数
    hidden_size = 128  # 隐藏层维度
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 30  # 增加训练轮次

    # 不同任务的序列长度设置
    pattern_lengths = []  # 模式检测任务
    adding_lengths = [400]  # Adding Problem
    copy_lengths = [ 120]  # Copy Memory Task

    noise_levels = []  # 噪声水平
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 运行实验
if __name__ == "__main__":
    print("=" * 60)
    print("Starting Memory Experiments")
    print(f"Device: {Config.device}")
    print("=" * 60)

    results = run_memory_experiments()
    visualize_results(results)

    # 保存结果
    torch.save(results, "memory_experiments_results.pth")
    print("Results saved to memory_experiments_results.pth")