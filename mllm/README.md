# Modern MLLM / vision-learning tutorials

顶层参考解和练习是小模型、CPU toy 的机制教学；`recipe/` 是真实数据、外部预训练权重和分布式训练所需的配方骨架。配置文件存在不表示已复现论文指标。

当前教程包括：

- `mae/`：masked-pixel reconstruction；`dino/`：DINO + iBOT 入门；`dinov2/`：独立heads、SK、KoLeo与逐样本iBOT；`dinov3/`：Gram anchoring、早期教师与高分辨率refinement；
- `ijepa/`：图像 context 到 EMA target latent 的预测；`vjepa/`：原始 V-JEPA 的视频 tubelet latent prediction；
- `blip2/`、`instructblip/`、`llava/`、`flamingo/`：视觉 token 接入语言模型的多模态训练链路；InstructBLIP 展示 instruction-aware Q-Former，Flamingo 展示交错多图文本与 gated cross-attention。
- `act/`：VLA / embodied 学习的基础视觉模仿策略；从 action chunk、CVAE 和 temporal ensemble 开始进入机器人控制。
- `openvla/`：原始 OpenVLA 的动作离散化、双视觉特征融合、动作 token 自回归训练与解码。
- `diffusion_policy/`：RSS 2023 条件动作扩散；CNN 时间 U-Net/FiLM 主线与完整 Transformer 扩展，共享 DDPM/DDIM、EMA 和滚动控制。
- `pi0/`：原始 π0（2024）的双流 VLM/action expert、分块注意力、flow matching、Euler 和 prefix K/V 缓存；可选接入真实 SmolVLM 256M。
- `backbones/`：[基础模型接口小课](backbones/README.md)，一次显式下载、按能力访问视觉/语言/KV；π0、LLaVA、OpenVLA 共用缓存与加载，默认 toy 仍离线。
- `repa/`、`rae/`：表征辅助的生成建模。[REPA](repa/README.md) 将干净教师 patch 特征对齐到生成器早期 hidden；[RAE](rae/README.md) 训练表征解码器，再在其高维 latent 上训练宽 head 生成器。两者独立学习，不需要预训练下载。

从仓库根目录安装共享基础包：

```bash
python3 -m pip install -e "general_utils/edu_core[dev]"
```

`general_utils/utils` 继续服务扩散/生成模型。MLLM 使用 `general_utils/edu_core` 提供的 `edu_core` token 语义；其中 `TubeletEmbed`、3D position interpolation 与 EMA schedule 由 I-JEPA/V-JEPA 等教程共享，避免重复实现且不把空间 GroupNorm attention 或 `[-1,1]` 图像约定混入 token 序列训练。

## 学习路径

先分清两个辅助目录：**foundations 教基础计算怎么实现；backbones 教现成模型怎么接入**。前者按需先学，后者在完成论文 toy 后选修；参考组件的实际实现位于 `general_utils/edu_core`，基础课填答不会自动替换该共享库。

先完成 [基础课程](foundations/README.md) 的 1（像素切片）、3（attention）、4（mask）、6（EMA）。DINO 另需 5（位置插值），V-JEPA 另需 2（tubelet）。基础练一次，后续论文默认调用共享组件。

|论文课程|进入前建议|关键练习|
|---|---|---|
|[DINO+iBOT](dino/README.md)|基础 1、3、4、5、6|CLS/patch、跨视图蒸馏、center、masked loss、teacher|
|[DINOv2](dinov2/README.md)|DINO+iBOT|block mask、双head、SK、加权iBOT、KoLeo、训练日程（六关）|
|[DINOv3 Gram](dinov3/README.md)|DINOv2|关系矩阵、早期快照、高分辨率对齐、两阶段训练（五关）|
|[I-JEPA](ijepa/README.md)|基础 1、3、4、6|矩形 mask、context/target、位置条件 predictor、latent regression|
|[V-JEPA](vjepa/README.md)|I-JEPA、基础 2|tubelet、时空 mask、完整视频 teacher|
|[LLaVA](llava/README.md)|基础 1、3、4|projector、占位符展开、answer labels、两阶段训练|
|[InstructBLIP](instructblip/README.md)|BLIP-2 或 query/cross-attention 基础|instruction-aware queries、query-only vision、双 token 输入、prefix|
|[Flamingo](flamingo/README.md)|基础 3、4，建议 LLaVA|Resampler、多图可见性、gate、LM 插层|
|[ACT](act/README.md)|基础 1、3、4；高斯分布均值/方差|chunk、CVAE、action decoder、temporal ensemble|
|[OpenVLA](openvla/README.md)|LLaVA；动作单位与量化|分位数归一化、action tokenizer、双视觉融合、labels/shift、自回归动作|
|[Diffusion Policy](diffusion_policy/README.md)|ACT 的 chunk 概念；高斯噪声；扩展另需 attention|轨迹窗口、动作扩散、FiLM/U-Net、DDPM/DDIM、EMA、滚动控制、Transformer masks|
|[π0](pi0/README.md)|OpenVLA、Diffusion Policy、基础 attention/mask；可选基础模型接口小课|双流QKV、state/action分块、flow速度目标、端到端梯度、Euler与KV缓存|
|[REPA](repa/README.md)|基础 1、3；EMA 可补基础 6；课程内补 flow/DiT|干净教师/带噪 latent、早期 hidden、负余弦、联合训练、无教师采样|
|[RAE](rae/README.md)|基础 1、3；课程内补 flow/DDT，无须先完成 REPA|冻结表征编码器、噪声增强 decoder、维度时间偏移、宽 head、两阶段训练|

表征辅助生成的路线：`foundations 1、3 → REPA 或 RAE 的课程内桥接 → practice 训练与采样 → 可选真实编码器/recipe`。REPA 提供额外训练监督；RAE 更换生成空间，默认没有 REPA loss。两课的冻结小编码器仅是机制 fixture，不等同于训练过的 DINO 表征。

推荐先学 DINO 或 LLaVA，再沿各自方向扩展。动作策略可按 ACT → Diffusion Policy / OpenVLA → π0 学习：从连续 chunk、迭代去噪和离散 token 自回归，进入预训练VLM与逐层动作专家。MAE、BLIP-2 保留原学习文件与已有填答；其运行方式以各目录说明为准，不纳入十二篇 practice 接口的统一验收。

## 每关怎么做

1. 阅读 lesson 的目标、前置、形状推导和手算例子。
2. 按编号补 TODO，不必先抄完整参考模型。
3. 卡住再读 HINTS.md，从思路到伪代码逐层展开。
4. 运行 `check_lessons.py --lesson N`，默认检查练习实现；省略关卡会检查全部。
5. 补完全部机制后，运行 demo 的 `--implementation practice`，检查自己的代码能否参与训练和推理。

以 LLaVA 为例，从仓库根目录执行：

```bash
PYTHONPATH=general_utils/edu_core python3 mllm/llava/check_lessons.py --lesson 1
PYTHONPATH=general_utils/edu_core python3 mllm/llava/run_llava_demo.py --implementation reference
PYTHONPATH=general_utils/edu_core python3 mllm/llava/run_llava_demo.py --implementation practice
```

参考模式用于验证答案；练习模式未完成会报告具体 TODO，不能自动切换到答案。

统一验收入口：

```bash
python3 mllm/check_tutorials.py --implementation reference
python3 mllm/check_tutorials.py --paper llava ijepa --implementation practice
python3 mllm/check_tutorials.py --checks wiring
python3 mllm/check_tutorials.py --paper openvla diffusion_policy --implementation reference
python3 mllm/check_tutorials.py --paper pi0 --implementation reference
python3 mllm/check_tutorials.py --paper repa rae --implementation reference
```

入口沿用当前 Python，并为子进程配置仓库共享包路径。Diffusion Policy 自动检查 CNN 和 Transformer；单独课程入口可用 `--backbone cnn|transformer|all`。`--checks wiring` 在测试中注入替身检查练习接线，不会填写或证明练习完成；默认 `all` 包含局部检查、demo 和接线测试。入口不下载权重或数据；任何入口失败都会返回非零退出码。当前验证环境和范围见 [验证记录](../docs/validation.md)，课程设计与进度模板见 [项目记忆](../docs/project_memory.md)。

REPA、RAE 使用包入口：`PYTHONPATH=general_utils/edu_core python3 -m mllm.repa.check_lessons --lesson 1`（RAE 将 `repa` 换成 `rae`）。demo 分别为 `mllm.repa.run_repa_demo`、`mllm.rae.run_rae_demo`，默认 reference，加 `--implementation practice` 才运行自己的完整填答。

真实小模型单独测试，不混入默认离线验收；下载与运行命令见 [接口小课](backbones/README.md)，整仓库依赖和能力边界见 [基础模型审查](../docs/backbone_audit.md)。

DINO系列按入门 → DINOv2 → DINOv3 Gram学习，保持旧入门目录和命令。新课使用包入口，从根执行 `PYTHONPATH=general_utils/edu_core python3 -m mllm.dinov2.check_lessons --lesson 1`；DINOv3同理。统一验收可选 `--paper dino dinov2 dinov3`。DINOv3是第4节Gram机制课，未包含完整新版架构或论文规模训练。
