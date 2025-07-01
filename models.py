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