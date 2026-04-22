import torch
import torch.nn as nn
from from_scratch.ddpm.ddpm import DDPM
from from_scratch.ddpm.datasets import get_img_shape,get_dataloader
from from_scratch.ddpm.model import (build_network,convnet_big_cfg,convnet_medium_cfg,convnet_small_cfg,unet_1_cfg,unet_res_cfg)
import cv2
import numpy as np
import einops

import dataclasses
import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
@dataclasses.dataclass
class Config:
    batch_size:int=512
    n_epochs:int=10


def train(ddpm:DDPM,config:Config,net,device,ckpt_path):
    n_steps=ddpm.n_steps
    dataloader=get_dataloader(config.batch_size)
    net=net.to(device)
    loss_fn=nn.MSELoss()
    optimizer=torch.optim.Adam(net.parameters(),1e-3)
    tic=time.time()
    for epoch in range(config.n_epochs):
        total_loss=0
        for x,_ in dataloader:
            crr_batch_size=x.shape[0]
            x=x.to(device)
            t=torch.randint(0,n_steps,(crr_batch_size,)).to(device)

            eps=torch.randn_like(x).to(device
                                       )
            
            xt=ddpm.sample_forward(x,t,eps)

            eps_theta=net(xt,t.reshape(crr_batch_size,1))
            loss=loss_fn(eps_theta,eps)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            total_loss+=loss.item()*crr_batch_size
        
        total_loss/=len(dataloader.dataset)
        toc=time.time()
        torch.save(net.state_dict(),ckpt_path)

        print(f'epoch: {epoch},loss: {total_loss}, time_cost: {(toc-tic):.2f}s ')
    

    print('Done! ')

def sample_imgs(ddpm,
                net,
                output_path,
                n_sample=81,
                device='cuda',
                simple_var=True):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28
        imgs = ddpm.sample_backward(shape,
                                    net,
                                    device=device,
                                    simple_var=simple_var).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(imgs,
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample**0.5))

        imgs = imgs.numpy().astype(np.uint8)

        cv2.imwrite(output_path, imgs)


configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]
def experiment_visualize_diff(ddpm, net, device, n_steps=1000, save_path='comparison.png'):
    net.eval()
    # 1. 准备一个固定的初始噪声 (1张图)
    shape = (1, *get_img_shape()) 
    x_init = torch.randn(shape).to(device)
    
    # 记录轨迹
    steps_to_show = np.linspace(999, 0, 20, dtype=int).tolist()
    traj_standard = []
    traj_mean_only = []

    with torch.no_grad():
        # --- 路径 A: 标准 DDPM (均值 + 噪声) ---
        xt = x_init.clone()
        for t in range(n_steps - 1, -1, -1):
            xt = ddpm.sample_backward_step(xt, t, net, simple_var=True)
            if t in steps_to_show:
                traj_standard.append(xt.cpu())

        # --- 路径 B: 纯均值 (只有 Mean) ---
        xt_m = x_init.clone()
        for t in range(n_steps - 1, -1, -1):
            # 模拟一个不加噪声的 step
            n = xt_m.shape[0]
            t_tensor = torch.tensor([t]*n).to(device).unsqueeze(1)
            eps = net(xt_m, t_tensor)
            
            # 计算均值 (这里直接用你代码里的公式)
            # 我们强制让最终返回的结果只有 mean，不加后面的 +noise
            x0 = (xt_m - torch.sqrt(1 - ddpm.alpha_bars[t]) * eps) / torch.sqrt(ddpm.alpha_bars[t])
            x0 = torch.clip(x0, -1, 1)
            mean = ddpm.coef1[t] * xt_m + ddpm.coef2[t] * x0
            
            xt_m = mean # 关键：不加噪声，直接把均值传给下一步
            if t in steps_to_show:
                traj_mean_only.append(xt_m.cpu())

    # --- 绘图逻辑 ---
    fig, axes = plt.subplots(2, len(steps_to_show), figsize=(15, 5))
    for i, (img_s, img_m) in enumerate(zip(traj_standard, traj_mean_only)):
        # 标准化到 [0, 1] 用于显示
        img_s = ((img_s + 1) / 2).clamp(0, 1).squeeze().numpy()
        img_m = ((img_m + 1) / 2).clamp(0, 1).squeeze().numpy()
        
        # 处理多通道/单通道显示
        if len(img_s.shape) == 3: # RGB
            img_s = img_s.transpose(1, 2, 0)
            img_m = img_m.transpose(1, 2, 0)

        axes[0, i].imshow(img_s, cmap='gray' if len(img_s.shape)==2 else None)
        axes[0, i].set_title(f"DDPM t={steps_to_show[i]}")
        axes[1, i].imshow(img_m, cmap='gray' if len(img_m.shape)==2 else None)
        axes[1, i].set_title(f"MeanOnly t={steps_to_show[i]}")
        
    for ax in axes.flatten(): ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"对比图已保存至: {save_path}")

if __name__ =='__main__':
    os.makedirs('work_dirs',exist_ok=True)

    n_steps=1000
    config_id=4
    device='cuda'
    model_path='from_scratch/ddpm/model_unet_res.pth'

    config=configs[config_id]
    net=build_network(config,n_steps)
    ddpm=DDPM(device,n_steps)
    total_para=sum(p.numel() for p in net.parameters())
    print(f'para total :{total_para:,}')
    # train_config = Config()
    # train(ddpm,train_config,net,device=device,ckpt_path=model_path)
    net=net.to(device)
    net.load_state_dict(torch.load(model_path))
    #sample_imgs(ddpm, net, 'work_dirs/diffusion.jpg', device=device)
    #experiment_visualize_diff(ddpm, net, device)