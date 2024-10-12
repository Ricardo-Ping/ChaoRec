"""
-*- coding: utf-8 -*-

@Author : Ricardo_PING
@Time : 2024/10/10 10:05
@File : DiffRec.py
@function :
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


class DNN(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type  # 时间步嵌入类型
        self.time_emb_dim = emb_size  # 时间步嵌入维度
        self.norm = norm  # 是否对输入进行归一化

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        # 将时间步嵌入与输入拼接
        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        # 定义输入层的线性层列表（用于逐层传播）
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        # 定义输出层的线性层列表
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                         for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        # 初始化输入层的权重和偏置
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]  # 输出维度
            fan_in = size[1]  # 输入维度
            std = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier 初始化标准差
            layer.weight.data.normal_(0.0, std)  # 权重正态分布初始化
            layer.bias.data.normal_(0.0, 0.001)  # 偏置正态分布初始化

        # 初始化输出层的权重和偏置
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        # 初始化嵌入层的权重和偏置
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        # 对时间步进行嵌入
        time_emb = self.timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        创建用于时间步的正弦嵌入。

        :param timesteps: 一个 1D 的 Tensor，包含每个批次元素的时间步。
        :param dim: 嵌入的维度。
        :param max_period: 控制嵌入的最小频率。
        :return: 一个 [N x dim] 形状的时间步嵌入张量。
        """

        half = dim // 2
        # 生成频率向量
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(timesteps.device)
        # 将时间步乘以频率，生成正弦和余弦嵌入
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # 如果嵌入维度为奇数，则补充一个零张量
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class DiffRec(nn.Module):
    def __init__(self, num_user, num_item, user_item_dict, noise_scale, noise_min,
                 noise_max, steps, dims, device):
        super(DiffRec, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.device = device

        self.noise_schedule = "linear-var"
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps

        self.mean_type = 'x0'  # x0, eps

        self.beta_fixed = True
        # 存储历史损失信息，每个步骤的历史损失存储数量为 `history_num_per_term`
        self.history_num_per_term = 10
        self.Lt_history = torch.zeros(steps, self.history_num_per_term, dtype=torch.float64).to(device)  # 每一步存储的历史损失
        self.Lt_count = torch.zeros(steps, dtype=torch.long).to(device)  # 每一步的计数器

        if noise_scale != 0.:  # 如果噪声缩放不为0，则生成 beta 序列
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(self.device)
            if self.beta_fixed:
                self.betas[0] = 0.00001  # 修复第一个 beta 的值，防止过拟合
            assert len(self.betas.shape) == 1, "betas 必须是一维的"
            assert len(self.betas) == self.steps, "beta 的数量必须等于扩散步骤数"
            # assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas 值超出范围"

        # 计算扩散过程所需的参数
        self.calculate_for_diffusion()

        # DNN
        self.dims = dims
        self.emb_size = 10
        self.norm = False
        out_dims = eval(self.dims) + [num_item]
        in_dims = out_dims[::-1]
        self.dnn = DNN(in_dims, out_dims, self.emb_size, time_type="cat", norm=self.norm).to(device)

    def calculate_for_diffusion(self):
        # 计算扩散过程中用到的一些中间变量
        alphas = 1.0 - self.betas  # alphas 是 1 - beta，表示信号保留的比例
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)  # 计算 alpha 的累积乘积，即 alpha_bar
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(
            self.device)  # alpha_{t-1}，前一个时间步的 alpha 累积值
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(
            self.device)  # alpha_{t+1}，后一个时间步的 alpha 累积值
        assert self.alphas_cumprod_prev.shape == (self.steps,), "alpha_cumprod_prev 的形状必须和步骤数一致"

        # 计算 sqrt(alpha_bar)，用于噪声采样时的计算
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # alpha_bar 的平方根
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # 1 - alpha_bar 的平方根
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)  # 1 - alpha_bar 的对数
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)  # alpha_bar 的倒数平方根
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)  # 1 / alpha_bar - 1 的平方根

        # 计算后验方差，用于在扩散过程中从 x_t 采样出 x_{t-1}
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # 对后验方差取对数，并确保它们不会超出合理范围，特别是时间步 1 的方差
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )

        # 计算后验均值的系数，用于预测 x_{t-1}
        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def get_betas(self):
        """
        根据噪声生成计划，创建扩散过程中的 beta 序列
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            # 如果噪声生成计划是线性或线性变体，则计算起始和结束噪声值
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                # 如果是线性生成计划，则使用 np.linspace 生成从 start 到 end 均匀分布的 self.steps 个 beta 值
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                # 如果是线性变体 (linear-var)，则调用 betas_from_linear_variance 方法生成 beta
                return self.betas_from_linear_variance(self.steps,
                                                       np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            # 如果噪声生成计划是余弦 (cosine)，则调用 betas_for_alpha_bar 方法来生成 beta
            return self.betas_for_alpha_bar(
                self.steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            )
        elif self.noise_schedule == "binomial":
            # 如果噪声生成计划是二项分布，则根据公式生成 beta 值
            # ts 是一个步数序列，betas 是一个递减序列，确保 beta 值逐渐减小
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")

    def betas_from_linear_variance(self, steps, variance, max_beta=0.999):
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])  # 将 beta 的第一个值设置为 1 - alpha_bar[0]
        for i in range(1, steps):
            # 在每一步中，计算 beta 值为 1 - alpha_bar[i] / alpha_bar[i - 1]
            # 其中 alpha_bar[i] 是当前步的 alpha 累积乘积，alpha_bar[i - 1] 是前一步的 alpha 累积乘积
            # 该公式确保每一步的 beta 值逐渐增大，但不超过 max_beta（默认为 0.999）
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
        return np.array(betas)

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        根据 alpha_bar 函数创建一个 beta 序列，该序列离散化了 alpha_bar 函数。
        alpha_bar 定义了 (1 - beta) 在时间 t = [0,1] 之间的累积乘积。

        :param num_diffusion_timesteps: 生成 beta 值的数量，即扩散的总步数。
        :param alpha_bar: 一个 lambda 函数，接受 t 作为参数，并生成 (1-beta) 的累积乘积。
                          该函数描述了在扩散过程中的信号保留比例。
        :param max_beta: beta 的最大值；通常设为小于1的值，以防止数值问题（例如奇异值）。
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            # t1 和 t2 分别是时间步的起始和结束时间，t 从 0 到 1 均匀分布
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def p_sample(self, model, x_start, steps, sampling_noise=False):
        """
            通过扩散模型进行采样，推断从 x_0 到 x_t 的过程。

            :param model: 模型，用于预测噪声或 x_0
            :param x_start: 初始输入 x_0
            :param steps: 推断的步骤数
            :param sampling_noise: 是否在采样过程中添加噪声
            :return: 经过多步扩散后生成的 x_t
            """
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            x_t = x_start
        else:
            # 将时间步 t 设为 steps - 1，用于在 q_sample 中进行采样
            t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            # 在 t 时刻从 x_start 采样出 x_t
            x_t = self.q_sample(x_start, t)

        # 将所有步骤从大到小排列，生成倒序的索引
        indices = list(range(self.steps))[::-1]

        # 如果噪声缩放为 0，则没有噪声，直接使用模型预测即可
        if self.noise_scale == 0.:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t  # 返回生成的 x_t

        # 正常的扩散过程，每一步都计算 x_t 并逐步推断到初始状态
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(model, x_t, t)
            # 如果启用了噪声采样，则在采样过程中添加随机噪声
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                # 通过均值和方差计算当前步的 x_t，添加噪声
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                # 不使用噪声，直接返回均值
                x_t = out["mean"]
        return x_t

    def q_sample(self, x_start, t, noise=None):
        """
            根据当前的时间步 t，结合噪声，生成在 t 时刻的扩散过程中的状态 x_t。

            :param x_start: 初始输入 x_0，即扩散过程的起点
            :param t: 当前的时间步，表示从 x_0 扩散到 x_t 的步数
            :param noise: 添加的噪声，如果未指定则随机生成
            :return: 扩散后的状态 x_t
            """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        计算扩散过程的后验均值和方差:
            q(x_{t-1} | x_t, x_0)

        :param x_start: 初始输入 x_0，即扩散过程的起点
        :param x_t: 当前时间步 t 的状态 x_t
        :param t: 当前时间步索引 t
        :return: 后验分布的均值和方差，以及对数方差
        """
        assert x_start.shape == x_t.shape
        # 后验均值是 x_0 和 x_t 的加权组合
        posterior_mean = (
                self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # 提取后验方差，用于确定从 x_t 推导到 x_{t-1} 的不确定性
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        # 提取后验对数方差，并进行裁剪，防止数值不稳定
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

    def p_mean_variance(self, model, x, t):
        """
            使用模型预测 p(x_{t-1} | x_t)，并预测初始状态 x_0。

            :param model: 扩散模型，用于预测噪声或 x_0
            :param x: 当前时间步 t 的状态 x_t
            :param t: 时间步 t
            :return: 返回均值、方差、对数方差以及预测的 x_0
            """
        # 获取 x 的 batch size 和通道数
        B, C = x.shape[:2]
        # 确保时间步 t 的形状与批量大小一致
        assert t.shape == (B,)
        # 使用模型生成输出，模型输入为 x_t 和时间步 t
        model_output = model(x, t)

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

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        从 1D 数组中提取值，并根据需要扩展为指定的形状。

        :param arr: 1D 张量（数组），它表示需要提取的参数（例如 sqrt_alphas_cumprod 或 sqrt_one_minus_alphas_cumprod）。
        :param timesteps: 时间步索引张量，表示从 arr 中提取的索引。
        :param broadcast_shape: 用于扩展的目标形状，通常是 [batch_size, 1, ...]。
        :return: 返回形状为 [batch_size, 1, ...] 的张量，其中包含 arr 对应时间步的值。
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def training_losses(self, model, x_start, reweight=False):
        """
            计算扩散模型的训练损失。

            :param model: 扩散模型，用于预测噪声或 x_0
            :param x_start: 初始输入 x_0
            :param reweight: 是否对不同时间步的损失进行重加权
            :return: 包含损失的字典
            """
        batch_size, device = x_start.size(0), x_start.device
        # 使用重要性采样的方法，随机选择每个样本的时间步 ts，并返回对应的采样权重 pt
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        # 生成与 x_start 相同形状的噪声
        noise = torch.randn_like(x_start)
        # 如果噪声缩放不为 0，则将噪声注入 x_t 中
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        model_output = model(x_t, ts)
        target = {
            'x0': x_start,
            'eps': noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        # 计算模型输出与目标之间的均方误差 (MSE)
        mse = self.mean_flat((target - model_output) ** 2)

        # 如果启用了重加权，则根据不同时间步对损失进行加权
        if reweight == True:
            if self.mean_type == 'x0':
                # 如果预测的是 x_0，使用信噪比 (SNR) 差异进行加权
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = torch.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == 'eps':
                # 如果预测的是噪声 epsilon，根据 alpha_cumprod 和 beta 进行加权
                weight = (1 - self.alphas_cumprod[ts]) / (
                        (1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts]))
                weight = torch.where((ts == 0), 1.0, weight)
                likelihood = self.mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output)) ** 2 / 2.0)
                loss = torch.where((ts == 0), likelihood, mse)
        else:
            # 如果没有加权，则损失权重为 1
            weight = torch.tensor([1.0] * len(target)).to(device)

        terms["loss"] = weight * loss

        # 更新 Lt_history 和 Lt_count（用于重要性采样）
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                # 当历史计数满时，将历史记录左移，丢弃最旧的损失
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                # 否则，记录新的损失
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError
        # 使用采样权重 pt 对损失进行归一化处理
        terms["loss"] /= pt
        return terms

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

    def mean_flat(self, tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

