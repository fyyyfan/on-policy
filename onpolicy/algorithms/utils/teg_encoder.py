import torch
import torch.nn as nn
from .util import init


class TEGCrossAgentAggregator(nn.Module):
    """
    跨智能体 TEG 特征聚合器 (MAPPO Critic 专用)。

    使用 Multi-head Self-Attention 捕捉智能体间 TEG 特征的交互关系，
    再通过均值池化压缩为固定维度表示，使输出维度与智能体数量 N 无关。

    输入: (batch, N_agents, embed_dim)
    输出: (batch, embed_dim)
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1, use_orthogonal=True):
        super().__init__()
        self.output_dim = embed_dim

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0),
                        gain=nn.init.calculate_gain('relu'))

        self.ffn = nn.Sequential(
            init_(nn.Linear(embed_dim, embed_dim * 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            init_(nn.Linear(embed_dim * 2, embed_dim))
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        for name, param in self.attn.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                init_method(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, agent_embeddings):
        """
        Args:
            agent_embeddings: (batch, N, embed_dim)
        Returns:
            aggregated: (batch, embed_dim) — 固定维度，与 N 无关
        """
        # (batch, N, embed_dim) -> (N, batch, embed_dim) for PyTorch < 1.9
        x_t = agent_embeddings.transpose(0, 1)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        attn_out = attn_out.transpose(0, 1)  # -> (batch, N, embed_dim)
        x = self.norm1(agent_embeddings + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.mean(dim=1)


class TEGEncoder(nn.Module):
    """
    Temporal Evolution Graph (TEG) 编码器。
    使用 GRU 对每个用户槽位的未来 K 步局部状态序列 x_τ 进行编码，
    提取最后一个时间步的隐状态 h_{t+K} 作为时序特征表示 h_TE。
    
    对于 num_user_slots 个用户槽位，共享同一个 GRU 分别编码，
    最终将各槽位的隐状态拼接为 (num_user_slots * hidden_size) 维向量。
    """

    def __init__(self, feature_dim, hidden_size, num_user_slots, K, use_orthogonal=True):
        super(TEGEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_user_slots = num_user_slots
        self.K = K
        self.output_dim = hidden_size * num_user_slots

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            batch_first=True
        )
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, teg_obs_flat):
        """
        Args:
            teg_obs_flat: (batch, num_user_slots * K * feature_dim)
        Returns:
            h_TE: (batch, num_user_slots * hidden_size)
        """
        batch_size = teg_obs_flat.size(0)
        # (batch, num_user_slots, K, feature_dim)
        teg = teg_obs_flat.view(batch_size, self.num_user_slots, self.K, self.feature_dim)
        # (batch * num_user_slots, K, feature_dim) — 共享 GRU 分别编码
        teg = teg.reshape(batch_size * self.num_user_slots, self.K, self.feature_dim)
        # GRU forward: h_n shape (1, batch*slots, hidden_size)
        _, h_n = self.gru(teg)
        h_n = h_n.squeeze(0)  # (batch * num_user_slots, hidden_size)
        h_n = self.norm(h_n)
        # (batch, num_user_slots * hidden_size)
        h_TE = h_n.reshape(batch_size, self.output_dim)
        return h_TE
