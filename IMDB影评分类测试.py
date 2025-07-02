import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random  # 新增random模块
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import math


# 设置全局随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 数据加载与预处理
class IMDBDataSet(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].split()[:self.max_length]
        sequence = [self.vocab.get(word, 0) for word in text]
        sequence = sequence + [0] * (self.max_length - len(sequence))
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def build_vocab(texts):
    vocab = {'<PAD>': 0}
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


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

        return self.out_proj(attn_output)  # 只取query对应部分


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


class CustomAttnRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.attn_rnn = AttnRNN(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.attn_rnn(x)  # 获取整个序列的输出
        output = self.fc(output[:, -1, :])  # 取最后一个时间步
        return output


class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.gru(x)
        output = self.fc(output[:, -1, :])
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead),
            num_layers
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, hidden_size)
        output = self.transformer_encoder(x)
        output = output[-1, :, :]  # Take the last time step
        output = self.fc(output)
        return output


# 训练与评估函数
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc


def main():
    # 设置全局随机种子
    set_seed(42)

    # 加载数据
    train_data = pd.read_csv('/kaggle/input/imdb-emotion-analysis-dataset/train.csv')
    test_data = pd.read_csv('/kaggle/input/imdb-emotion-analysis-dataset/test.csv')

    label_mapping = {'neg': 0, 'pos': 1}  # 假设 'neg' 表示负面，'pos' 表示正面
    train_data['label'] = train_data['label'].map(label_mapping)
    test_data['label'] = test_data['label'].map(label_mapping)

    train_texts = train_data['text'].values
    train_labels = train_data['label'].values
    test_texts = test_data['text'].values
    test_labels = test_data['label'].values

    # 构建词汇表
    vocab = build_vocab(train_texts)
    input_size = len(vocab)
    hidden_size = 128
    output_size = 2  # 二分类
    max_length = 200
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集和数据加载器
    train_dataset = IMDBDataSet(train_texts, train_labels, vocab, max_length)
    test_dataset = IMDBDataSet(test_texts, test_labels, vocab, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型
    models = {
        'AttnRNN': CustomAttnRNN(input_size, hidden_size, output_size),  # 新增自制模型
        'VanillaRNN': VanillaRNN(input_size, hidden_size, output_size),
        'LSTM': LSTM(input_size, hidden_size, output_size),
        'GRU': GRU(input_size, hidden_size, output_size),
        'Transformer': TransformerModel(input_size, hidden_size, output_size)
    }

    # 训练和评估模型
    criterion = nn.CrossEntropyLoss()
    for name, model in models.items():
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print(f'Training {name}...')
        acc = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device)
        print(f'{name} Test Accuracy: {acc:.4f}')


if __name__ == '__main__':
    main()