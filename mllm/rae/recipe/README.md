# 从CPU机制到真实RAE实验

此目录是外部复现路线与配置合同，**没有真实ImageNet trainer、权重加载器、LPIPS/GAN实现或FID评估器**。顶层随机encoder不是预训练表示，L1不是论文完整重构目标。配置通过仅表示metadata自洽；不把下载权重、真实训练或指标复现记为完成。

1. 按[原论文v1 §3/C](https://arxiv.org/html/2510.11690v1)和[官方仓库](https://github.com/bytetriper/RAE)准备ImageNet-1K训练/验证集、类别标签及训练许可，记录版本、数据清单与预处理。实际数据路径由用户配置，未验证本机存在。
2. 显式获取官方DINOv2-with-registers及RAE配置/权重，记录revision、校验和、encoder与decoder结构。去掉CLS/register与LN affine；DINOv2输入224、patch14产生16×16 tokens，decoder patch16输出256图像。不得把toy权重当官方兼容模型。
3. 先训练decoder：冻结encoder，保留论文L1+LPIPS+GAN、自适应对抗权重、判别器结构及分阶段启用时刻。LPIPS需要真实感知网络权重，GAN有独立判别器更新，不能用随机特征距离冒名替代。v1幅度为半正态；当前官方main为uniform，选用前必须明确版本与配方，不能无声混合。原论文C.2有Adam/学习率/批量/epoch表。
4. 冻结RAE后训练DiT/DDT：256tokens×768维，latent patch_size=1，reference_dim=4096。按官方config设置宽head、类别条件dropout、EMA、优化器、time分布与分布式；CPU课的简单time embedding/LayerNorm/GELU需换回所选官方架构。真实ImageNet规模需要加速器和分布式训练，未在本课测量资源或时间。
5. 评估重构rFID与生成gFID分开进行，使用论文的ImageNet划分、50K样本、Euler步数和相同特征统计。CFG/AutoGuidance需要额外策略/模型；不能拿默认无条件toy与论文最佳指标直接比较。保存encoder/decoder/generator/EMA、归一化、时间合同和随机种子。

从仓库根验证metadata：

```bash
python3 -m mllm.rae.recipe.validate_config
```

`training_executed=false`、`paper_metrics_reproduced=false`是当前证据边界；若将来实际运行，另建带日期/命令/资源/评估协议的结果记录，不通过修改布尔值制造证据。
