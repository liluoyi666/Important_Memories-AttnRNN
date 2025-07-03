# AttnRNN: 高效门控注意力RNN模型

## 项目简介

AttnRNN 是一种融合注意力机制与门控机制的创新型循环神经网络（RNN）模型。其核心思想是让当前历史信息与新输入信息直接竞争注意力分数，结合残差连接和极简门控设计，有效缓解梯度消失，提升长序列记忆能力和收敛速度。AttnRNN 在多个经典序列学习任务上实现了对传统RNN、GRU和LSTM的跨时代性能超越。

---

## 模型结构

### EfficientAttention

- 仅对查询Q进行线性投影，K/V直接采用原始context，避免冗余参数
- 不做掩码，序列长度固定，提升效率
- 多头注意力，输出仅与query对应

**核心代码片段：**
```python
class EfficientAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        ...
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, query, context):
        ...
        q = self.q_proj(query)
        k = context
        v = context
        ...
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        ...
        return self.out_proj(attn_output)
```

### AttnRNNCell

- 融合投影、注意力、单一门控、残差和层归一化
- “历史记忆+新输入”共同参与注意力竞争，门控调节信息流

### AttnRNN

- 结构简洁，仅需一个隐藏状态
- 提供与传统 RNN/GRU/LSTM 兼容的接口

---

## 参数量与复杂度对比

| 模型       | 参数量公式                     | d=h=64, h=128 示例参数量 |
|------------|-------------------------------|-------------------------|
| RNN        | dh + h² + 2h                  | 24,832                  |
| GRU        | 3(dh + h² + h)                | 74,496                  |
| LSTM       | 4(dh + h² + h)                | 99,328                  |
| AttnRNN    | 4h² + 5h (d=h) 或 dh+4h²+6h    | 74,496                  |

- 参数量与 GRU 同级，远小于 LSTM

**时间复杂度**（单步）：
- RNN:        O(B(dh + h²))
- GRU:        O(3B(dh + h²))
- LSTM:       O(4B(dh + h²))
- **AttnRNN:**  O(B(3h² + dh))

> 注：Python for 循环未优化时，AttnRNN 实测耗时约为 GRU 的 2.5~3.5 倍，但理论主项系数更优。

---

## 梯度消失缓解能力

| 序列长度 | AttnRNN    | RNN        | LSTM       | GRU        |
|----------|------------|------------|------------|------------|
| 16       | 3.26e-2    | 1.02e-7    | 4.89e-4    | 9.97e-4    |
| 128      | 1.52e-3    | 0          | 1.75e-10   | 4.86e-16   |
| 352      | 3.27e-5    | 0          | 0          | 0          |
| 512      | 2.28e-6    | 0          | 0          | 0          |

> **AttnRNN 在超长序列下梯度依然不消失，领先传统RNN模型十亿量级以上。**

---

## 核心实验结果

### 1. Adding Problem（加法记忆）
平均绝对误差 MAE（越低越好）：

| 长度  | RNN   | GRU   | LSTM  | AttnRNN |
|-------|-------|-------|-------|---------|
| 50    | 0.34  | 0.011 | 0.067 | 0.0069  |
| 200   | 0.32  | 0.0048| 0.023 | 0.0014  |
| 400   | 0.35  | 0.021 | 0.043 | 0.0011  |

### 2. Copy Memory Task（复制记忆）
准确率 Acc（越高越好）：

| 长度  | RNN   | GRU   | LSTM  | AttnRNN |
|-------|-------|-------|-------|---------|
| 30    | 0.138 | 0.662 | 0.548 | 0.814   |
| 120   | 0.164 | 0.346 | 0.301 | 0.438   |

### 3. IMDB 情感分类任务
最高准确率：

| 模型         | 参数量   | 最高准确率 |
|--------------|----------|------------|
| Vanilla RNN  | 33,024   | 0.5072     |
| GRU          | 99,072   | 0.7840     |
| LSTM         |132,096   | 0.7744     |
| **AttnRNN**  | 66,176   | 0.8016     |
| Transformer  |1,186,048 | 0.8144     |

> **AttnRNN 以极少参数量逼近 transformer 性能，远优于其他类RNN。**

---

## 设计与创新点

- **注意力机制**：历史状态与新输入直接竞争注意力分数，捕获关键信息，滤除噪声。
- **残差连接**：确保历史信息主导，缓解梯度消失。
- **极简门控**：仅需单一门控层，既补全长时记忆又减少参数。
- **隐藏空间**：只需单一隐藏状态，存储开销低。

---

## 总结

AttnRNN 代表了一种高效、易用、易于扩展的 RNN 新范式，具备：

- **极强的长序列记忆能力**：梯度不消失，历史信息充分保留；
- **高效参数结构**：参数量与 GRU 同级，远小于 LSTM；
- **快速收敛**：在多任务中以极快速度达到最优性能；
- **理论与实验证据齐备**：在加法、复制记忆、文本分类等任务中均大幅超越传统RNN模型。

**更多思考**  
本项目源自于对生物神经系统的启发，即“此时此刻新旧信息的竞争性融合”，将 transformer 的部分机制融入 RNN 框架，赋予其更强的序列处理能力。

---

## 引用与联系方式

项目地址：[https://github.com/liluoyi666/Important_Memories-AttnRNN](https://github.com/liluoyi666/Important_Memories-AttnRNN)

如有建议或合作意向，欢迎 issue 或 PR！
