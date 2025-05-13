"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/10/12 22:03
@File : CF_Diff.py
@function :
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CAM_AE(nn.Module):
    """
    CAM-AE: 该神经网络架构用于在扩散模型的逆向过程中学习数据分布。
    一跳邻居（直接邻居）信息将会被集成。
    """

    def __init__(self, d_model, num_heads, num_layers, in_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(CAM_AE, self).__init__()
        self.in_dims = in_dims
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.num_layers = num_layers

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        # 输入层和输出层
        self.in_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip([d_model, d_model], [d_model, d_model])])
        self.out_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip([d_model, d_model], [d_model, d_model])])
        # 多层前馈层
        self.forward_layers = nn.ModuleList([nn.Linear(d_model, d_model) for i in range(num_layers)])
        # 内部维度定义，用于编码用户-物品交互
        self.dim_inters = 1024

        # 一跳邻居嵌入和解码层
        self.first_hop_embedding = nn.Linear(1, d_model)  # Expend dimension
        self.first_hop_decoding = nn.Linear(d_model, 1)

        # 二跳邻居嵌入层
        self.second_hop_embedding = nn.Linear(1, d_model)  # Expend dimension

        # 输出层，结合时间嵌入和内部特征
        self.final_out = nn.Linear(self.dim_inters + emb_size, self.dim_inters)

        # Dropout层，用于防止过拟合
        self.drop = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(0.8)  # 第一层 Dropout
        self.drop2 = nn.Dropout(dropout)  # 第二层 Dropout

        # 自编码器的编码器和解码器
        self.encoder = nn.Linear(self.in_dims, self.dim_inters)  # 编码用户-物品交互数据
        self.decoder = nn.Linear(self.dim_inters + emb_size, self.in_dims)  # 解码层，将数据映射回输入维度
        self.encoder2 = nn.Linear(self.in_dims, self.dim_inters)

        # 注意力层
        self.self_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.5, batch_first=True)
            for i in range(num_layers)
        ])

        self.time_emb_dim = emb_size
        self.d_model = d_model

        # LayerNorm层，用于标准化输入和输出
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, x_sec_hop, timesteps):
        """
        前向传播过程
        :param x: 输入的一跳邻居数据
        :param x_sec_hop: 输入的二跳邻居数据
        :param timesteps: 时间步，用于生成时间嵌入
        """

        # Step 1: 编码一跳和二跳邻居信息
        x = self.encoder(x)  # 对一跳邻居数据进行编码
        h_sec_hop = self.encoder(x_sec_hop)  # 对二跳邻居数据进行编码

        # Step 2: 生成时间步嵌入
        time_emb = self.timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)

        # Step 3: 如果设置了归一化，则对输入进行归一化
        if self.norm:
            x = F.normalize(x)

        # Step 4: 添加 Dropout 进行正则化
        x = self.drop(x)

        # Step 5: 将一跳邻居信息和时间嵌入拼接
        h = torch.cat([x, emb], dim=-1)
        h = h.unsqueeze(-1)  # 增加一维度
        h = self.first_hop_embedding(h)  # 对一跳信息进行扩展维度

        # Step 6: 处理二跳邻居信息
        h_sec_hop = torch.cat([h_sec_hop, emb], dim=-1)
        h_sec_hop = h_sec_hop.unsqueeze(-1)  # 增加一维度
        h_sec_hop = self.second_hop_embedding(h_sec_hop)  # 对二跳信息进行扩展维度

        # Step 7: 多层自注意力机制
        for i in range(self.num_layers):
            attention_layer = self.self_attentions[i]  # 选择当前层的多头自注意力层
            attention, attn_output_weights = attention_layer(h_sec_hop, h, h)  # 计算注意力

            # 添加注意力结果并进行残差连接
            attention = self.drop1(attention)
            h = h + attention
            # h = self.norm1(h)

            # 第二层 Dropout
            h = self.drop2(h)

            # 前馈层
            forward_pass = self.forward_layers[i]
            h = forward_pass(h)

            # 使用 tanh 激活函数
            if i != self.num_layers - 1:
                h = torch.tanh(h)

        # Step 8: 解码一跳信息
        h = self.first_hop_decoding(h)
        h = torch.squeeze(h)  # 去掉多余的维度
        h = torch.tanh(h)
        h = self.decoder(h)  # 解码成输出

        return h

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class CF_Diff(nn.Module):
    def __init__(self, num_user, num_item, user_item_dict, noise_scale, noise_min,
                 noise_max, steps, learning_rate, device):
        super(CF_Diff, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.device = device
        self.learning_rate = learning_rate

        self.noise_schedule = "linear"
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.sampling_steps = 0
        self.sampling_noise = False

        self.reweight = True  # reweight the loss for different timesteps
        if self.noise_scale == 0.0:
            self.reweight = False

        self.mean_type = 'x0'  # x0, eps

        self.beta_fixed = True
        # 存储历史损失信息，每个步骤的历史损失存储数量为 `history_num_per_term`
        self.history_num_per_term = 10
        self.Lt_history = torch.zeros(steps, self.history_num_per_term, dtype=torch.float64).to(device)  # 每一步存储的历史损失
        self.Lt_count = torch.zeros(steps, dtype=torch.long).to(device)  # 每一步的计数器

        if noise_scale != 0.0:  # 如果噪声缩放不为0，则生成 beta 序列
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(self.device)
            if self.beta_fixed:
                self.betas[0] = 0.00001  # 修复第一个 beta 的值，防止过拟合
            assert len(self.betas.shape) == 1, "betas 必须是一维的"
            assert len(self.betas) == self.steps, "beta 的数量必须等于扩散步骤数"
            # assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas 值超出范围"

        # 计算扩散过程所需的参数
        self.calculate_for_diffusion()

        # CAM_AE
        self.emb_size = 10
        # d_model, num_heads, num_layers, in_dims, emb_size
        self.CAM_AE = CAM_AE(16, 4, 2, num_item, self.emb_size).to(device)

    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return self.betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            return self.betas_for_alpha_bar(
                self.steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            )
        elif self.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")

    def betas_from_linear_variance(self, steps, variance, max_beta=0.999):
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
        return np.array(betas)

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                          produces the cumulative product of (1-beta) up to that
                          part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                         prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(
            self.device)  # alpha_{t-1}
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(
            self.device)  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def mean_flat(self, tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_sample(self, x_start, x_sec_hop):
        steps = self.sampling_steps
        assert steps <= self.steps, "Too much steps in inference."
        #print("inference step:", steps)
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = self.CAM_AE(x_t, x_sec_hop, t)
            return x_t

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(x_t, x_sec_hop, t)
            if self.sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t

    def p_mean_variance(self, x, x_sec_hop, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = self.CAM_AE(x, x_sec_hop, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        if self.mean_type == 'x0':
            pred_xstart = model_output
        elif self.mean_type == 'eps':
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)

        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def training_losses(self, x_start, x_sec_hop):

        batch_size, device = x_start.size(0), x_start.device
        # 使用重要性采样的方法，随机选择每个样本的时间步 ts，并返回对应的采样权重 pt
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        # 生成与 x_start 相同形状的噪声
        noise = torch.randn_like(x_start)
        # 如果噪声缩放不为 0，则将噪声注入 x_t 中
        if self.noise_scale != 0.0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        model_output = self.CAM_AE(x_t, x_sec_hop, ts)
        target = {
            'x0': x_start,
            'eps': noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        # 计算模型输出与目标之间的均方误差 (MSE)
        mse = self.mean_flat((target - model_output) ** 2)

        reloss = self.reweight_loss(x_start, x_t, mse, ts, target, model_output, device)
        self.update_Lt_history(ts, reloss)

        # importance sampling
        reloss /= pt
        mean_loss = reloss.mean()

        return mean_loss

    def reweight_loss(self, x_start, x_t, mse, ts, target, model_output, device):
        weight = 0
        loss = 0
        if self.reweight:
            if self.mean_type == 'x0':
                # Eq.11
                weight = self.SNR(ts - 1) - self.SNR(ts)
                # Eq.12
                weight = torch.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == "eps":
                weight = (1 - self.alphas_cumprod[ts]) / (
                        (1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts])
                )
                weight = torch.where((ts == 0), 1.0, weight)
                likelihood = self.mean_flat(
                    (x_start - self._predict_xstart_from_eps(x_t, ts, model_output))
                    ** 2
                    / 2.0
                )
                loss = torch.where((ts == 0), likelihood, mse)
        else:
            weight = torch.tensor([1.0] * len(target)).to(device)
            loss = mse
        reloss = weight * loss
        return reloss

    def update_Lt_history(self, ts, reloss):
        # update Lt_history & Lt_count
        for t, loss in zip(ts, reloss):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        """
           根据采样方法，选择用于训练的时间步 ts，并计算每个时间步的采样权重 pt。

           :param batch_size: 需要采样的时间步数量（通常等于批量大小）
           :param device: 设备信息（CPU 或 GPU）
           :param method: 采样方法，默认为 'uniform'，可以选择 'importance' 或 'uniform'
           :param uniform_prob: 在重要性采样时，使用均匀采样的概率
           :return: 采样的时间步 ts 及其对应的采样权重 pt
           """
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')

            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, dim=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt

        elif method == 'uniform':  # uniform sampling
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()

            return t, pt

        else:
            raise ValueError
