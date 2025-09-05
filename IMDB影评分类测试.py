import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import os


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


from models import AttnRNN


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


# 训练与评估函数（修改为返回训练损失和测试准确率列表）
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, epochs=10, model_name="model"):
    model.to(device)

    # 记录训练过程
    train_losses = []
    test_accuracies = []

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

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 评估测试集
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
        test_accuracies.append(acc)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Test Acc: {acc:.4f}')

    return train_losses, test_accuracies


def main():
    # 设置全局随机种子
    set_seed(42)

    # 加载数据
    train_data = pd.read_csv(TRAIN_DATA)
    test_data = pd.read_csv(TEST_DATA)

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
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集和数据加载器
    train_dataset = IMDBDataSet(train_texts, train_labels, vocab, max_length)
    test_dataset = IMDBDataSet(test_texts, test_labels, vocab, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型
    models = {
        'AttnRNN': CustomAttnRNN(input_size, hidden_size, output_size),
        'VanillaRNN': VanillaRNN(input_size, hidden_size, output_size),
        'LSTM': LSTM(input_size, hidden_size, output_size),
        'GRU': GRU(input_size, hidden_size, output_size),
        'Transformer': TransformerModel(input_size, hidden_size, output_size)
    }

    # 训练和评估模型，收集训练过程数据
    criterion = nn.CrossEntropyLoss()
    results = {}
    train_losses_dict = {}
    test_accuracies_dict = {}

    for name, model in models.items():
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print(f'Training {name}...')
        train_losses, test_accuracies = train_and_evaluate(
            model, train_loader, test_loader, criterion, optimizer, device, epochs, name)

        train_losses_dict[name] = train_losses
        test_accuracies_dict[name] = test_accuracies
        results[name] = test_accuracies[-1]
        print(f'{name} Final Test Accuracy: {results[name]:.4f}')

    # 创建目录保存图表
    os.makedirs("training_plots", exist_ok=True)

    # 绘制合并的训练损失曲线
    plt.figure(figsize=(10, 6))
    for name, losses in train_losses_dict.items():
        plt.plot(range(1, epochs + 1), losses, label=name)

    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_plots/combined_training_loss.png')
    plt.close()

    # 绘制合并的测试准确率曲线
    plt.figure(figsize=(10, 6))
    for name, accs in test_accuracies_dict.items():
        plt.plot(range(1, epochs + 1), accs, label=name)

    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_plots/combined_test_accuracy.png')
    plt.close()

    # 打印最终结果比较
    print("\nModel Comparison:")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")


if __name__ == '__main__':
    print("请在[https://www.kaggle.com/datasets/hengjianshe/imdb-emotion-analysis-dataset]下载数据集并解压\n")

    TRAIN_DATA="archive/train.csv"
    TEST_DATA = "archive/test.csv"


    main()