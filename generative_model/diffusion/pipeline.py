"""DDPM math scaffold.

Implement the TODO methods yourself in this order:
1. beta schedule and alpha_bar buffers
2. q_sample
3. epsilon-prediction training loss
4. p_sample (one reverse DDPM step)
5. sample (the reverse loop)
"""

import torch
from torch.nn import functional as F


class DDPMPipeline:
    """Discrete variance-preserving diffusion with epsilon prediction."""

    def __init__(
        self,
        num_train_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        self.num_train_steps = num_train_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # TODO 1:
        # Create the following 1-D tensors with shape (T,):
        # self.betas
        # self.alphas = 1 - betas
        # self.alpha_bars = cumulative product of alphas
        #
        # Keep them on CPU here. Move the gathered coefficients to x.device
        # inside helper methods.
        self.betas = torch.linspace(beta_start,beta_end,num_train_steps)
        self.alphas = 1-self.betas
        self.alpha_bars = torch.cumprod(self.alphas,dim=0)
        # \bar{alpha}_{-1}=1 is needed by q(x_{t-1} | x_t, x_0).
        self.alpha_bars_prev = torch.cat([torch.ones(1), self.alpha_bars[:-1]])

    def _extract(self, values, timesteps, x):
        """Gather values[t] and reshape to (B,1,1,1) for broadcasting.

        TODO 2:
        - values is a (T,) schedule tensor
        - timesteps is integer (B,)
        - return values[timesteps] reshaped for x: (B,C,H,W)
        """
        # Move before indexing: CUDA indices cannot index a CPU tensor.
        out = values.to(device=x.device)[timesteps]
        out = out.to(dtype=x.dtype)
        return out.reshape(-1,1,1,1)

    def q_sample(self, x0, timesteps, noise=None):
        """Forward process q(x_t | x_0).

        TODO 3:
        x_t = sqrt(alpha_bar_t) * x0
            + sqrt(1 - alpha_bar_t) * noise

        Return x_t and the exact noise used.
        """
        sqrt_alpha_bar_t=torch.sqrt(
            self._extract(self.alpha_bars,timesteps,x0)
        )
        sqrt_1_alphabar_t=torch.sqrt(1-self._extract(self.alpha_bars,timesteps,x0))
        if noise is None:
            # DDPM's forward process uses standard Gaussian epsilon.
            noise = torch.randn_like(x0)
        
        x_t=sqrt_alpha_bar_t*x0+sqrt_1_alphabar_t*noise

        return x_t,noise

    def training_loss(self, model, x0):
        """Epsilon-prediction objective.

        TODO 4:
        1. sample integer t in [0, T)
        2. call q_sample(x0, t)
        3. epsilon_pred = model(x_t, t.float())
        4. loss = MSE(epsilon_pred, noise)

        Return: loss, {"t": t.float().mean()}
        """
        batch_size=x0.shape[0]
        device=x0.device
        t=torch.randint(0,self.num_train_steps,(batch_size,),device=device)
        x_t,noise=self.q_sample(x0,t)

        noise_pred=model(x_t,t.float())

        loss=F.mse_loss(noise_pred,noise)

        return loss, {"t": t.float().mean()}

    @torch.no_grad()
    def p_sample(self, model, x_t, timesteps):
        """One stochastic DDPM reverse step: x_t -> x_{t-1}.

        TODO 5:
        Derive the posterior mean from epsilon_pred and alpha schedules.
        Add posterior noise only when t > 0.

        Inputs:
            x_t: (B,C,H,W)
            timesteps: integer (B,), all values usually identical in sampling
        """
        noise_pred=model(x_t,timesteps.float())

        alpha_t=self._extract(self.alphas,timesteps,x_t)
        alpha_bar_t=self._extract(self.alpha_bars,timesteps,x_t)
        beta_t=self._extract(self.betas,timesteps,x_t)
        alpha_bar_prev_t = self._extract(self.alpha_bars_prev, timesteps, x_t)
        mu_theta=(1/torch.sqrt(alpha_t))*(x_t-(beta_t/torch.sqrt(1-alpha_bar_t))*noise_pred)
        beta_t_tilda=beta_t*(1-alpha_bar_prev_t)/(1-alpha_bar_t)
        noise=torch.randn_like(x_t)

        # At t=0, q(x_{-1}|x_0) is deterministic: no extra noise.
        nonzero_mask = (timesteps != 0).to(x_t.dtype).view(-1, 1, 1, 1)
        x_prev = mu_theta + nonzero_mask * torch.sqrt(beta_t_tilda) * noise

        return x_prev

    @torch.no_grad()
    def sample(self, model, batch_size, image_shape, device):
        """Start from Gaussian noise and call p_sample from T-1 down to 0.

        TODO 6:
        x = torch.randn(batch_size, *image_shape, device=device)
        for step in reversed(range(self.num_train_steps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x
        """
        
        x=torch.randn(batch_size,*image_shape,device=device)

        for step in reversed(range(self.num_train_steps)):
            t=torch.full(
                (batch_size,),
                step,
                device=device,
                dtype=torch.long
            )
            x=self.p_sample(model,x,t)
        return x
