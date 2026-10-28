# 从机制toy到原始REPA实验

本目录提供metadata与官方实验路线，**没有实现本地真实数据trainer**。配置检查通过不表示数据已下载、训练已执行或FID已复现。

1. 获取ImageNet许可并准备官方256px训练数据。原图与预计算VAE moments必须严格同序、同裁剪，类别ID必须一致。
2. 按[官方README](https://github.com/sihyun-yu/REPA)准备Stable Diffusion VAE与DINOv2-B/14；固定官方代码commit、checkpoint来源和依赖版本。encoder常用sd-vae-ft-mse，论文附录D说明decode使用sd-vae-ft-ema，不可混为本toy codec。
3. VAE latent为4×32×32，SiT patch2得到16×16 tokens；DINOv2将RGB调整为224，移除CLS/只取x_norm_patchtokens，并调整位置编码匹配16×16。变换坐标与空间次序应逐例检查，不以插值特征掩盖错位。
4. SiT-B/2第4层对齐；L/XL第8层。三层SiLU projector、negative patch cosine、lambda=.5、linear路径与velocity、训练teacher no_grad、denoiser EMA。按官方数据预处理保留RGB与moments配对；无需用额外随机crop破坏配对。
5. 论文附录D报告batch256、AdamW lr1e-4且无weight decay，使用8×H100 80GB；这是官方资源报告，不是本课的运行需求或资源保证。正式分布式、AMP、恢复与CFG使用官方训练系统。
6. 评估采用明确的checkpoint、50K样本、ADM evaluator与参考统计；原论文主结果使用250步Euler-Maruyama SDE，含最后0.04步处理。toy Euler ODE不能直接对应这些FID。分别报告生成质量与representation linear probe/CKNNA，随机toy可视化不替代这些指标。

```bash
python3 -m mllm.repa.recipe.validate_config
python3 -m mllm.repa.recipe.validate_config --require-data
```

第二条在默认数据路径缺失时应失败。后续真实执行另记录日期、硬件、版本、训练步和评估结果；本配置默认`training_executed=false`、`paper_metrics_reproduced=false`。尚未提供真实VAE/DINO加载器、ImageNet预处理、DDP trainer、FID/CKNNA评估器；不要把配置骨架称为已完成能力。

来源：[原论文附录D/E](https://arxiv.org/html/2410.06940v3#A4)、[官方实现](https://github.com/sihyun-yu/REPA)。无默认自动下载。
