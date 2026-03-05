import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.teg_encoder import TEGEncoder, TEGCrossAgentAggregator
from onpolicy.utils.util import get_shape_from_obs_space


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        obs_dim = obs_shape[0]

        # TEG 编码器配置
        self._use_teg = getattr(args, 'prediction_window_K', 0) > 0
        if self._use_teg:
            teg_K = args.prediction_window_K
            teg_feature_dim = getattr(args, 'teg_feature_dim', 9)
            teg_num_user_slots = getattr(args, 'teg_num_user_slots', 2)
            teg_hidden_size = getattr(args, 'teg_hidden_size', 32)
            self._teg_obs_dim = teg_num_user_slots * teg_K * teg_feature_dim
            self._base_obs_dim = obs_dim - self._teg_obs_dim

            self.teg_encoder = TEGEncoder(
                teg_feature_dim, teg_hidden_size,
                teg_num_user_slots, teg_K, self._use_orthogonal)

            self.base = MLPBase(args, (self._base_obs_dim,))

            # 融合层：将 MLP 特征与 TEG 编码拼接后映射回 hidden_size
            fusion_input_dim = self.hidden_size + self.teg_encoder.output_dim
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
            active_func = nn.ReLU() if getattr(args, 'use_ReLU', True) else nn.Tanh()
            gain = nn.init.calculate_gain('relu' if getattr(args, 'use_ReLU', True) else 'tanh')

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

            self.teg_fusion = nn.Sequential(
                init_(nn.Linear(fusion_input_dim, self.hidden_size)),
                active_func,
                nn.LayerNorm(self.hidden_size)
            )
        else:
            base = CNNBase if len(obs_shape) == 3 else MLPBase
            self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)
        self.algo = args.algorithm_name

    def _process_obs(self, obs):
        """将 obs 拆分为 base_obs 和 teg_obs 分别处理，融合后返回 actor_features。"""
        if self._use_teg:
            base_obs = obs[..., :self._base_obs_dim]
            teg_obs = obs[..., self._base_obs_dim:]
            base_features = self.base(base_obs)
            teg_features = self.teg_encoder(teg_obs)
            return self.teg_fusion(torch.cat([base_features, teg_features], dim=-1))
        else:
            return self.base(obs)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self._process_obs(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self._process_obs(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.algo == "hatrpo":
            action_log_probs, dist_entropy ,action_mu, action_std, all_probs= self.act.evaluate_actions_trpo(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu"),
                 per_agent_obs_dim=None):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        cent_obs_dim = cent_obs_shape[0]

        self._use_teg = getattr(args, 'prediction_window_K', 0) > 0
        use_centralized_V = getattr(args, 'use_centralized_V', True)

        # MAPPO + TEG: Critic 自有 TEG 编码器 + 跨智能体注意力聚合
        self._use_teg_mappo = (self._use_teg and use_centralized_V
                               and per_agent_obs_dim is not None)
        # IPPO + TEG: Critic 独立 TEG 编码器（单智能体视角）
        self._use_teg_ippo = self._use_teg and not use_centralized_V

        if self._use_teg_mappo:
            teg_K = args.prediction_window_K
            teg_feature_dim = getattr(args, 'teg_feature_dim', 9)
            teg_num_user_slots = getattr(args, 'teg_num_user_slots', 2)
            teg_hidden_size = getattr(args, 'teg_hidden_size', 32)
            teg_agg_heads = getattr(args, 'teg_agg_num_heads', 4)
            teg_agg_dropout = getattr(args, 'teg_agg_dropout', 0.1)

            self._per_agent_teg_dim = teg_num_user_slots * teg_K * teg_feature_dim
            self._per_agent_obs_dim = per_agent_obs_dim
            self._per_agent_base_dim = per_agent_obs_dim - self._per_agent_teg_dim
            self._num_agents = cent_obs_dim // per_agent_obs_dim

            self.teg_encoder = TEGEncoder(
                teg_feature_dim, teg_hidden_size,
                teg_num_user_slots, teg_K, self._use_orthogonal)

            self.teg_aggregator = TEGCrossAgentAggregator(
                self.teg_encoder.output_dim, teg_agg_heads,
                teg_agg_dropout, self._use_orthogonal)

            base_input_dim = self._num_agents * self._per_agent_base_dim
            self.base = MLPBase(args, (base_input_dim,))

            fusion_input_dim = self.hidden_size + self.teg_aggregator.output_dim
            active_func = nn.ReLU() if getattr(args, 'use_ReLU', True) else nn.Tanh()
            gain = nn.init.calculate_gain('relu' if getattr(args, 'use_ReLU', True) else 'tanh')

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

            self.teg_fusion = nn.Sequential(
                init_(nn.Linear(fusion_input_dim, self.hidden_size)),
                active_func,
                nn.LayerNorm(self.hidden_size)
            )
        elif self._use_teg_ippo:
            teg_K = args.prediction_window_K
            teg_feature_dim = getattr(args, 'teg_feature_dim', 9)
            teg_num_user_slots = getattr(args, 'teg_num_user_slots', 2)
            teg_hidden_size = getattr(args, 'teg_hidden_size', 32)
            self._teg_obs_dim = teg_num_user_slots * teg_K * teg_feature_dim
            self._base_obs_dim = cent_obs_dim - self._teg_obs_dim

            self.teg_encoder = TEGEncoder(
                teg_feature_dim, teg_hidden_size,
                teg_num_user_slots, teg_K, self._use_orthogonal)

            self.base = MLPBase(args, (self._base_obs_dim,))

            fusion_input_dim = self.hidden_size + self.teg_encoder.output_dim
            active_func = nn.ReLU() if getattr(args, 'use_ReLU', True) else nn.Tanh()
            gain = nn.init.calculate_gain('relu' if getattr(args, 'use_ReLU', True) else 'tanh')

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

            self.teg_fusion = nn.Sequential(
                init_(nn.Linear(fusion_input_dim, self.hidden_size)),
                active_func,
                nn.LayerNorm(self.hidden_size)
            )
        else:
            base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
            self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def _process_obs(self, cent_obs):
        """处理 Critic 输入，根据模式选择不同的 TEG 处理管线。"""
        if self._use_teg_mappo:
            batch = cent_obs.shape[0]
            # (batch, N, per_agent_obs_dim)
            agent_obs = cent_obs.view(batch, self._num_agents, self._per_agent_obs_dim)

            base_obs = agent_obs[..., :self._per_agent_base_dim]
            teg_obs = agent_obs[..., self._per_agent_base_dim:]

            base_features = self.base(base_obs.reshape(batch, -1))

            teg_flat = teg_obs.reshape(batch * self._num_agents, -1)
            teg_encoded = self.teg_encoder(teg_flat)
            teg_encoded = teg_encoded.view(batch, self._num_agents, -1)
            agg_teg = self.teg_aggregator(teg_encoded)

            return self.teg_fusion(torch.cat([base_features, agg_teg], dim=-1))
        elif self._use_teg_ippo:
            base_obs = cent_obs[..., :self._base_obs_dim]
            teg_obs = cent_obs[..., self._base_obs_dim:]
            base_features = self.base(base_obs)
            teg_features = self.teg_encoder(teg_obs)
            return self.teg_fusion(torch.cat([base_features, teg_features], dim=-1))
        else:
            return self.base(cent_obs)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self._process_obs(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
