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

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, context):
        batch_size, q_len = query.size(0), query.size(1)

        # 投影Q并重塑多头
        q = self.q_proj(query)
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 处理KV
        k = context.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = k  # 使用相同值作为V

        # 注意力计算
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

        return self.out_proj(attn_output)


class AttnRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads

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
        # h_prev: [batch, q_len, hidden_size]
        # x: [batch, input_size]

        x_proj = self.proj_layer(x)  # [batch, hidden_size]

        # 构建上下文
        x_exp = x_proj.unsqueeze(1)  # [batch, 1, hidden_size]

        # 四部分上下文
        part1 = h_prev  # [batch, q_len, H]
        part2 = x_exp  # [batch, 1, H]
        part3 = h_prev + x_exp  # 广播加法
        part4 = h_prev * x_exp  # 广播乘法

        # 拼接KV [batch, 3*q_len + 1, H]
        context = torch.cat([part1, part2, part3, part4], dim=1)

        # 注意力计算
        attn_out = self.attn(query=h_prev, context=context)
        attn_out = F.gelu(attn_out)

        # 门控机制
        gate_input = torch.cat([h_prev, attn_out], dim=-1)  # [batch, q_len, 2H]
        update_gate = self.update_gate(gate_input)  # [batch, q_len, H]

        # 残差连接 + 层归一化
        h_candidate = self.layer_norm(attn_out + h_prev)

        # 门控更新
        h_new = update_gate * h_candidate + (1 - update_gate) * h_prev
        return h_new


class AttnRNN2d(nn.Module):
    def __init__(self, input_size, hidden_size, q_len=4, batch_first=True, init_value=0.001):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_len = q_len  # 隐藏状态序列长度
        self.batch_first = batch_first
        self.init_value = init_value
        self.cell = AttnRNNCell(input_size, hidden_size)

    def forward(self, x, h0=None):
        # 统一为batch_first模式
        if not self.batch_first:
            x = x.permute(1, 0, 2)  # [seq_len, batch, features] -> [batch, seq_len, features]

        batch_size, seq_len, _ = x.shape

        # 初始化隐藏状态 [batch, q_len, hidden_size]
        if h0 is not None:
            h = h0
        else:
            h = torch.full(
                (batch_size, self.q_len, self.hidden_size),
                self.init_value,
                device=x.device,
                dtype=x.dtype
            )

        # 存储压缩后的输出向量
        output_vectors = []
        # 存储完整的隐藏状态（用于返回最终状态）
        hidden_states = []

        for t in range(seq_len):
            h = self.cell(h, x[:, t, :])
            # 压缩隐藏状态：在q_len维度求和 -> [batch, hidden_size]
            compressed_output = h.sum(dim=1)
            output_vectors.append(compressed_output)
            hidden_states.append(h)

        # 输出形状 [batch, seq_len, hidden_size]
        output = torch.stack(output_vectors, dim=1)

        # 最终隐藏状态（保持多维形状）
        final_hidden = hidden_states[-1]

        # 恢复时间优先格式（如果需要）
        if not self.batch_first:
            output = output.permute(1, 0, 2)  # [seq_len, batch, hidden_size]

        return output, final_hidden

