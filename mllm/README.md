# Modern MLLM / vision-learning tutorials

顶层参考解和练习是小模型、CPU toy 的机制教学；`recipe/` 是真实数据、外部预训练权重和分布式训练所需的配方骨架。配置文件存在不表示已复现论文指标。

当前教程包括：

- `mae/`：masked-pixel reconstruction；`dino/`：DINO + iBOT 的无标签 self-distillation；
- `ijepa/`：图像 context 到 EMA target latent 的预测；`vjepa/`：原始 V-JEPA 的视频 tubelet latent prediction；
- `blip2/`、`instructblip/`、`llava/`、`flamingo/`：视觉 token 接入语言模型的多模态训练链路；InstructBLIP 展示 instruction-aware Q-Former，Flamingo 展示交错多图文本与 gated cross-attention。
- `act/`：VLA / embodied 学习的基础视觉模仿策略；从 action chunk、CVAE 和 temporal ensemble 开始进入机器人控制。

从仓库根目录安装共享基础包：

```bash
python3 -m pip install -e "general_utils/edu_core[dev]"
```

`general_utils/utils` 继续服务扩散/生成模型。MLLM 使用 `general_utils/edu_core` 提供的 `edu_core` token 语义；其中 `TubeletEmbed`、3D position interpolation 与 EMA schedule 由 I-JEPA/V-JEPA 等教程共享，避免重复实现且不把空间 GroupNorm attention 或 `[-1,1]` 图像约定混入 token 序列训练。

## 学习路径

先完成 [基础课程](foundations/README.md) 的 1（像素切片）、3（attention）、4（mask）、6（EMA）。DINO 另需 5（位置插值），V-JEPA 另需 2（tubelet）。基础练一次，后续论文默认调用共享组件。

|论文课程|进入前建议|关键练习|
|---|---|---|
|[DINO+iBOT](dino/README.md)|基础 1、3、4、5、6|CLS/patch、跨视图蒸馏、center、masked loss、teacher|
|[I-JEPA](ijepa/README.md)|基础 1、3、4、6|矩形 mask、context/target、位置条件 predictor、latent regression|
|[V-JEPA](vjepa/README.md)|I-JEPA、基础 2|tubelet、时空 mask、完整视频 teacher|
|[LLaVA](llava/README.md)|基础 1、3、4|projector、占位符展开、answer labels、两阶段训练|
|[InstructBLIP](instructblip/README.md)|BLIP-2 或 query/cross-attention 基础|instruction-aware queries、query-only vision、双 token 输入、prefix|
|[Flamingo](flamingo/README.md)|基础 3、4，建议 LLaVA|Resampler、多图可见性、gate、LM 插层|
|[ACT](act/README.md)|基础 1、3、4；高斯分布均值/方差|chunk、CVAE、action decoder、temporal ensemble|

推荐先学 DINO 或 LLaVA，再沿各自方向扩展。MAE、BLIP-2 保留原学习文件与已有填答；其运行方式以各目录说明为准，不纳入七篇新 practice 接口的统一验收。

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
```

入口沿用当前 Python，并为子进程配置仓库共享包路径。`--checks wiring` 在测试中注入替身检查练习接线，不会填写或证明练习完成；默认 `all` 包含局部检查、demo 和接线测试。入口不下载权重或数据；任何入口失败都会返回非零退出码。当前验证环境和范围见 [验证记录](../docs/validation.md)，课程设计与进度模板见 [项目记忆](../docs/project_memory.md)。
