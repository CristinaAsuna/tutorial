import torch
import numpy as np

class DDPMSampler:

    def __init__(self, generator: torch.Generator, 
                 num_training_steps=1000, 
                 beta_start: float = 0.00085,
                   beta_end: float = 0.0120):
        # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        # For the naming conventions, refer to the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator

        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self,num_inference_steps=50):
        self.num_inference_steps=num_inference_steps
        step_ratio=self.num_train_timesteps//self.num_inference_steps

        timesteps=(np.arange(0,num_inference_steps)*step_ratio).round()[::-1].copy()

        self.timesteps=torch.from_numpy(timesteps)


    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t=timestep-self.num_train_timesteps//self.num_inference_steps

        return prev_t
    
    def set_strength(self, strength=1):
        start_step=self.num_inference_steps-int(self.num_inference_steps*strength)
        self.timesteps=self.timesteps[start_step:]
        self.start_step=start_step

    #forward
    """
    q(xt|x0)=N(xt; sqrt(alpha_bar)*x0,(1-sqrt(alpha_bar))*I)
    x0-->origin img
    I-->noise (0,1)
    """

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        alpha_bart=self.alphas_cumprod.to(device=original_samples.device,dtype=original_samples.dtype)

        timesteps=timesteps.to(original_samples.device)

        #sqrt(alpha_bar)
        sqrt_alpha_bar=alpha_bart[timesteps]**0.5
        sqrt_alpha_bar=sqrt_alpha_bar.flatten()

        #
        while len(sqrt_alpha_bar.shape)<len(original_samples.shape):
            sqrt_alpha_bar=sqrt_alpha_bar.unsqueeze(-1)
        # the equation is varance,we need std
        sqrt_one_minus_alpha_bar=(1-alpha_bart[timesteps])**0.5
        sqrt_one_minus_alpha_bar=sqrt_one_minus_alpha_bar.flatten()
        while len(sqrt_one_minus_alpha_bar.shape)<len(original_samples.shape):
            sqrt_one_minus_alpha_bar=sqrt_one_minus_alpha_bar.unsqueeze(-1)

        #noise I
        noise=torch.randn(original_samples.shape,generator=self.generator,
                          device=original_samples.device,
                          dtype=original_samples.dtype)
        
        noise_samples=sqrt_alpha_bar*original_samples+sqrt_one_minus_alpha_bar*noise

        return noise_samples
    
    #caculate the var of revserse process
    def _get_variance(self, timestep: int) -> torch.Tensor:
        """
        beta_tildat=(1-alpha_bar_prevt)/(1-alpha_bar_t)*beta_t

        这里的t是离散的
        所以beta_t,无法再使用原来连续的继续算,而是对alpha_bar_t和alpha_bar_t-1进行变形,
        得到beta_t的其他表示方法

        """
        prev_t=self._get_previous_timestep(timestep)
        alpha_bar_t=self.alphas_cumprod[timestep]

        alpha_bar_prev_t=self.alphas_cumprod[prev_t] if prev_t >=0 else self.one

        beta_t=1-alpha_bar_t/alpha_bar_prev_t
        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample

        var=(1-alpha_bar_prev_t)/(1-alpha_bar_t)*beta_t

        var=torch.clamp(var,min=1e-20)

        return var
    

    def step(self, timestep: int, 
             latents: torch.Tensor, 
             model_output: torch.Tensor):
        
        t=timestep
        prev_t=self._get_previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        one_minus_alpha_bar = 1 - alpha_prod_t
        one_minus_alpha_bar_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        #x0 which was predict
        #x0=(xt-sqrt(1-alpha_bart)*noise_unet(xt))/sqrt(alpha_bart)
        x0=(latents-one_minus_alpha_bar**(0.5))*model_output/alpha_prod_t**(0.5)

        """
        mean=sqrt(alpha_bar_prev)*beta_t/(1-aplha_bar_t)*x0 
        
        + sqrt(alpha_t)*(1-alpha_bar_prev)/(1-alpha_bar)*xt


        
        """
        drif=(alpha_prod_t_prev**(0.5)*current_beta_t)/one_minus_alpha_bar
        diff_coeff=current_alpha_t**(0.5)*one_minus_alpha_bar_prev/one_minus_alpha_bar

        mean=drif*x0+diff_coeff*latents

        #add noise
        var=0
        if t>0:
            device = model_output.device
            noise = torch.randn(model_output.shape, 
                                generator=self.generator, device=device, dtype=model_output.dtype)
            var=(self._get_variance(t)**0.5)*noise
        
        prev_img=mean+var
        return prev_img

