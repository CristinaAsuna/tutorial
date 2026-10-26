# 从机制 toy 到真实 OpenVLA

本目录提供**配置合同与外部官方实验路线**，没有本地 RLDS trainer、LoRA 实现、权重适配器或 LIBERO runner。`validate_config.py` 只检查元数据，不会训练、下载权重或启动硬件。

## 1. 先固定数据与动作语义

起点选 [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) 的官方兼容 RLDS 数据。使用 [OpenVLA 官方仓库](https://github.com/openvla/openvla) 的安装、数据标准化和微调说明；将其独立 checkout 并记录 commit，环境与本课程 CPU 环境分开。读取官方 `prismatic/vla/datasets/rlds/oxe/` 中对应 dataset 的标准化函数，不凭 7 维长度推断单位。

每个训练 step 至少提供主相机 RGB、自然语言指令和标准化后的 7 维动作。平移/旋转是数据 adapter 统一后的末端控制，夹爪按该数据集转换。数据按照 episode 分训练/验证，不能用验证轨迹拟合 q01/q99。保存统计 mask、维度顺序、数据集名称和 `unnorm_key`；夹爪是否归一化必须核对 adapter，示例配置采用前六维归一化、夹爪不归一化。

本 toy 的 `prompts` 是已知整数列表。真实路线使用官方 Llama tokenizer、prompt builder、图像 transform 与 BOS/EOS；不要把 toy 的 320 词表或 8×8 RGB 传给真实 checkpoint。

## 2. 从官方权重开始真实微调

权重来源：[openvla/openvla-7b 模型卡](https://huggingface.co/openvla/openvla-7b)。遵循其依赖和加载代码，按当前模型卡确定精度、显存与可访问的许可条件。toy 架构不能 `load_state_dict` 官方权重。

在官方 checkout 中先查看 `vla-scripts/finetune.py --help`，按照官方 README 设置 `vla_path`、`data_root_dir`、`dataset_name`、输出目录和实际硬件的 batch/累积步数。示例 metadata 默认 LoRA 微调路线，原论文的全参数训练作为另一条明确路线；它们不等同于本课程随机初始化小模型的一次 optimizer step。

硬件与资源：小模型只需 CPU；7B 模型须另配满足模型卡要求的 GPU 环境。先完成一批真实数据的 forward/backward 与 action-token 解码检查，再扩大训练。实际显存、吞吐、可用 GPU 数量在本机未测量；不能由 toy 耗时外推。若从视觉/语言预训练权重开始论文规模 Open-X 混合训练，需要完整数据 mixture 和分布式资源，超出本 recipe 本地实现。

记录 checkpoint、tokenizer、图像预处理、数据 split/mixture、归一化统计、LoRA/full 模式和训练参数。更换数据集时重新选择匹配的统计键，不静默用 BridgeData 统计解码其他机器人的动作。

## 3. 评估分两层

离线先评估 action-token accuracy 和反归一化后的逐维动作误差；报告夹爪与连续动作分别的结果。高 token accuracy 不等于机器人任务成功。

模拟环境路线使用官方 LIBERO 微调/评估说明和**相应 LIBERO checkpoint**；原始 BridgeData checkpoint 与 LIBERO 数据动作语义不同，不能直接宣称可评估。按官方任务集、初始状态、种子、训练数据和每任务 rollout 次数报告 success rate，保存失败案例和 checkpoint 选择规则。这里未安装或执行 LIBERO。

## 元数据检查

从教程仓库根运行：

```bash
python3 mllm/openvla/recipe/validate_config.py
python3 mllm/openvla/recipe/validate_config.py --config mllm/openvla/recipe/config.json
```

相对数据路径按当前工作目录解析。默认路径是占位符，`dataset exists: False` 是明确的缺失提示；配置通过只证明合同满足，不证明 RLDS schema、数据存在或真实训练成功。
