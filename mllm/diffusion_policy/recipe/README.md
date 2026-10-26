# 从 CPU 动作扩散到 Push-T / robomimic

本目录是**外部官方实验路线与可校验配置骨架**。没有本地 Push-T trainer、Zarr loader、模拟环境或论文 checkpoint loader；metadata 检查不代表真实训练或任务复现。

## 1. 从一个真实数据集开始

使用 [官方仓库](https://github.com/real-stanford/diffusion_policy) 提供的 Push-T 数据、安装环境与训练说明，先固定 checkout commit。不要把整个官方依赖栈装进本教程的最小 CPU 环境。示例配置选择图像 Push-T 的 replay Zarr 路线。

数据逻辑为每 episode 的图像、状态和动作序列，以及明确的 episode ends。Push-T 动作是二维绝对位置。按 episode 分训练/验证，训练统计缩放连续输入与动作，图像使用官方 crop/normalize；首尾窗口重复 padding，窗口不跨 episode。模型接收历史 To 步观测，动作窗口从 `t-To+1` 开始。

先只读一条真实轨迹：人工核对相机顺序、图像值域、动作单位和观测/动作对齐；输出一个窗口及 `To-1:To-1+Ta` 的执行部分。不应仅检查形状。真实状态归一化须由官方 normalizer 处理，本 toy 的 4 维合成 state 不是 Push-T schema。

## 2. CNN 起步，再替换 Transformer

在官方独立 checkout 中查看 README 和 `train.py --help`；选择 [train_diffusion_unet_hybrid_workspace 配置](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_hybrid_workspace.yaml)以及匹配的 Push-T image task，设置真实数据路径与输出目录。网络规模、ResNet/SpatialSoftmax、图像裁剪、EMA schedule、optimizer 和训练日程以该固定版本官方配置为准。

之后换成 [train_diffusion_transformer_hybrid_workspace](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_transformer_hybrid_workspace.yaml)，保留相同数据 split、动作单位和评估条件，重新按其配置调参。这里没有将配置 JSON 自动转换为 Hydra，也没有提供可冒充真实训练的 stub 命令。示例 `Tp=16,To=2,Ta=8` 是 CNN 起点；论文 Transformer 某些任务采用不同 horizon，复现时使用对应论文/官方任务配置并记录差异。

本课程 CNN 主线选 FiLM 条件，原论文 Push-T 某些 CNN 报告采用 inpainting；若要精确对应那一行结果，必须使用其条件配置。不能把当前 FiLM toy 直接叫作该指标的复现。

噪声训练默认 100 步 square-cosine/epsilon objective，标准 DDPM 推理；DDIM 加速时明确记录推理步数与 eta。观察特征必须在每次规划中只计算一次。真实资源路线先在 GPU 上测一个 batch 的显存/吞吐，再设 batch、训练步数和保存频率；本机没有测量真实训练资源，也不把 toy 16 步推断延迟外推成控制频率。

权重使用官方项目页/仓库提供的 checkpoint 与配套训练配置，记录下载来源和文件 hash。toy 的小 U-Net/Transformer 与官方 checkpoint 不兼容。

## 3. 指标与迁移

先报告 held-out 噪声损失，并在官方 Push-T 环境运行 rollout，采用其 target area coverage 指标。区分最后 checkpoint 与依据验证集选出的 best checkpoint；记录种子、初始状态、rollout 次数、执行 horizon 与控制频率。固定噪声可重复不证明动作分布已经学好；合成 MSE 不等于成功率。

迁移到 robomimic 时使用官方任务数据 adapter 和配置，重新明确动作表示、相机/低维观测、归一化与环境 metric；不能把 Push-T 二维 schema 直接扩展几个维度就宣称复现。论文评估包含跨种子、多初态与 checkpoint 汇总，正式实验应按所选任务的原协议执行。

## 检查配置合同

从教程仓库根执行：

```bash
python3 mllm/diffusion_policy/recipe/validate_config.py
python3 mllm/diffusion_policy/recipe/validate_config.py --config mllm/diffusion_policy/recipe/config.json
```

配置检查 horizon、两种网络、epsilon 目标、噪声日程、EMA 和统计范围。数据路径按当前工作目录解析；不存在时明确输出 `False`。配置通过不证明真实数据、外部依赖、权重或 rollout 已验证。

证据：[论文 §II–III、实验与附录](https://arxiv.org/html/2303.04137v1)、[项目主页与公开数据/权重](https://diffusion-policy.cs.columbia.edu/)、[官方代码](https://github.com/real-stanford/diffusion_policy)。
