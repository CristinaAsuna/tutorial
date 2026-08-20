import torch
from torch.nn import functional as F

import math
class VPSDE:
    """
    VP-SDE:

    dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dW

    t ∈ [eps, 1]
    """

    def __init__(
        self,
        beta_min=0.1,
        beta_max=20.0,
        eps=1e-5,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = eps

    def beta(self, t):
        """
        t: (B,)
        return: (B,)

        TODO:
        实现线性 beta schedule：
        beta(t) = beta_min + t * (beta_max - beta_min)
        """
        return self.beta_min+ t*(self.beta_max-self.beta_min)
        

    def marginal_prob(self, x0, t):
        """
        给定原始数据 x0，返回 q(x_t | x0) 的均值和标准差。

        x0: (B, C, H, W)
        t:  (B,)

        return:
            mean: (B, C, H, W)
            std:  (B,)

        TODO:
        根据 VP-SDE 推导 alpha(t)、sigma(t)，使得：

        x_t = alpha(t) * x0 + sigma(t) * noise
        """

        # B(t) = ∫_0^t beta(s) ds
        t=t.float().to(x0.device)
        beta_t=self.beta_min*t+0.5*(self.beta_max-self.beta_min)*t**2

     # alpha(t) = exp(-0.5 * B(t))
        alpha_t=torch.exp(-0.5*beta_t)



     # sigma(t) = sqrt(1 - exp(-B(t)))
        # 用 -expm1(-B(t)) = 1 - exp(-B(t))，数值更稳
        sigma_t=torch.sqrt(-torch.expm1(-beta_t))


        # alpha: (B,) -> (B, 1, 1, 1)
        mean=alpha_t[:,None,None,None].to(x0.dtype)*x0
        std=sigma_t

        return mean,std



    def sde(self, x, t):
        """
        原始正向 SDE 的 drift 与 diffusion。

        return:
            drift:    (B, C, H, W)
            diffusion: (B,)

        TODO:
        drift = -0.5 * beta(t) * x
        diffusion = sqrt(beta(t))
        """
        beta_t=self.beta(t)
        drift=-0.5*beta_t[:,None,None,None]*x
        diffusion=torch.sqrt(beta_t)

        return drift,diffusion


class ScoreMatchingPipeline:
    """
    模型约定：

    score_pred = model(x_t, model_time)

    score_pred 的 shape 与 x_t 相同，
    其含义是 score: ∇_{x_t} log p_t(x_t)
    """

    def __init__(self, sde:VPSDE, time_scale=1000.0):
        self.sde = sde
        self.time_scale = time_scale

    def model_time(self, t):
        """将连续 t ∈ [0,1] 映射到 U-Net time embedding 的尺度。"""
        return t * self.time_scale

    def make_training_pair(self, x0):
        """
        return:
            x_t:          加噪状态，(B,C,H,W)
            t:            连续时间，(B,)
            noise:        高斯噪声，(B,C,H,W)
            score_target: (B,C,H,W)
            std:          (B,)
        """
        batch_size = x0.shape[0]
        device=x0.device

        # TODO 1：从 [eps, 1] 均匀采样 t
        # t = ...
        t=torch.rand(batch_size,device=device)*(1-self.sde.eps)+self.sde.eps


        # TODO 2：采样 epsilon
        # noise = ...
        noise=torch.randn_like(x0)

        # TODO 3：调用 self.sde.marginal_prob(x0, t)
        # mean, std = ...
        mean,std=self.sde.marginal_prob(x0,t)

        # TODO 4：构造 x_t
        # std_view = std[:, None, None, None]
        # x_t = mean + std_view * noise

        std_view=std[:,None,None,None]

        x_t=mean+std_view*noise

        # TODO 5：构造 score target
        # score_target = -noise / std_view

        score_target=-noise/std_view

        return x_t,t,noise,score_target,std

    def training_loss(self, model, x0):
        """
        建议使用加权 denoising score matching：

        || sigma(t) * score_theta(x_t, t) + noise ||²

        它比直接拟合 -noise/sigma 更稳定。
        """

        # TODO：
        # x_t, t, noise, score_target, std = self.make_training_pair(x0)
        # score_pred = model(x_t, self.model_time(t))
        # std_view = std[:, None, None, None]
        # loss = F.mse_loss(std_view * score_pred, -noise)
        x_t,t,noise,score_target,std=self.make_training_pair(x0)
        std_view=std[:,None,None,None]

        score_pred=model(x_t,self.model_time(t))

        loss=F.mse_loss(std_view*score_pred,-noise)

        return loss, {"t": t.detach().mean(), "std": std.detach().mean()}

    @torch.no_grad()
    def sample_euler_maruyama(
        self,
        model,
        batch_size,
        image_shape,
        steps,
        device,
    ):
        """
        reverse SDE 的 Euler-Maruyama sampler。

        从 t=1 的噪声出发，走到 t=eps。

        TODO：
        1. x = N(0, I)
        2. 构造从 1 到 eps 的时间网格
        3. 每一步：
           - score = model(x, model_time(t))
           - drift, diffusion = self.sde.sde(x, t)
           - reverse_drift = drift - diffusion² * score
           - x = x + reverse_drift * dt
                 + diffusion * sqrt(abs(dt)) * random_noise
        """
        x=torch.randn(batch_size,*image_shape,device=device)

        dt=(self.sde.eps-1.0)/steps

        for i in range(steps):
            t=1.0+i*dt

            t_tensor=torch.full(
                (batch_size,),
                t,
                device=device,
                dtype=x.dtype,

            )
            score=model(x,self.model_time(t_tensor))

            drift,diffusion=self.sde.sde(x,t_tensor)

            #reverse
            diffusion_sq=(diffusion**2)[:,None,None,None]
            reverse_drift=drift-diffusion_sq*score

            x=x+reverse_drift*dt

            noise=torch.randn_like(x)
            x=x+diffusion[:,None,None,None]*math.sqrt(abs(dt))*noise
        return x
