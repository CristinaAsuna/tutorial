# RAE：让表示编码器成为生成模型的latent空间

本课实现 Boyang Zheng、Nanye Ma、Shengbang Tong、Saining Xie 的 **Diffusion Transformers with Representation Autoencoders**，2025，[arXiv:2510.11690v1](https://arxiv.org/html/2510.11690v1)，采用原始版本，非后续RAEv2/T2I。它复用冻结表示编码器、训练图像decoder，再在高维表示上训练生成器。**没有VAE posterior或KL，也没有REPA辅助对齐loss。**两课共享的是利用视觉表示改善生成的研究问题，表示的角色不同。

目标是亲手写出机制与两阶段CPU闭环；默认不下载模型或数据，不声称随机fixture具有预训练语义，不声称生成图像具有论文质量。

## 先修与入口

[foundations](../foundations/README.md) 是基础算法练习：先完成1（patch重排）、3（attention）、6（EMA）。本课第5关提供Flow Matching桥接，无须预学或先完成REPA；[REPA](../repa/README.md) 仅供可选比较，本课不导入它的代码。[backbones](../backbones/README.md) 是取用现成模型组件的选修课，本课不要求下载SmolVLM；后续真实RAE需要视觉表示模型，不需要语言模型。

全部命令在仓库根目录、已安装torch的环境运行：

```bash
PYTHONPATH=general_utils/edu_core python3 -m mllm.rae.run_rae_demo
PYTHONPATH=general_utils/edu_core python3 -m mllm.rae.check_lessons --lesson 1
# 填完一关后换编号；全部填完再运行整模型：
PYTHONPATH=general_utils/edu_core python3 -m mllm.rae.run_rae_demo --implementation practice
PYTHONPATH=.:general_utils/edu_core python3 -m pytest -q mllm/rae/tests
```

默认demo为reference；默认局部检查为practice。未完成报告具体函数并返回2，真实语义错误直接失败；产品没有答案回退。单关只有选中关使用学生函数，其余参考外围用于隔离错误。整模型practice的编码、decoder、loss、两次训练、EMA控制及采样均调用学生ops。维护测试注入替身证明路由，并不代表学生填答完成。

## 两阶段数据流

```text
阶段1：images B,3,16,16
  → frozen encoder → [CLS,REG,patches] B,18,64
  → 剥离2个特殊tokens → 无affine逐token LN → B,16,64
  → 训练时latent半正态幅度噪声 → ViT decoder → 每patch48个像素
  → unpatchify → B,3,16,16 → L1 → 只更新decoder

阶段2：冻结整个RAE
  → clean latent tokens ↔ grid B,64,4,4
  → shifted time、Gaussian noise → x_t B,16,64
  → 窄base B,16,32 → 逐token条件 B,16,96
  → 宽head B,16,96（独立读取原始x_t）→ velocity B,16,64
  → flow loss → 只更新生成器 → optimizer之后更新EMA

采样：noise → EMA + shifted reverse Euler(1→0) → latent → frozen decoder → RGB
```

|关卡|亲手实现|关键手算/检查|
|---|---|---|
|1 encoding|剥离特殊token、无affine LN、tokens/grid转换|[1,3]→[-1,1]；编号patch保持宽度最快|
|2 decoder|ViT输出映射及unpatchify|16个4×4×3块→16×16 RGB|
|3 augmentation|每样本半正态sigma和逐元素Gaussian扰动|tau=.2,normal=-2,eps=3,z=2→3.2|
|4 reconstruction|L1与stage1更新|[1,3]对[0,1]→1.5；encoder不更新|
|5 flow|维度shift、插值、速度loss|m/n=4,t=.5→2/3；z=2,eps=0,t=.25→1.5,-2|
|6 DDT|窄base、逐tokenAdaLN条件、独立宽噪声路径|32→96条件，原始64维输入另投到96，不经过32瓶颈|
|7 training|stage2更新与EMA|旧EMA2、更新后student6、m=.75→3|
|8 sampling|偏移网格Euler、完整图像生成|x=0,v=-2，从1积分到0→2|

每关有模块骨架、编号步骤、形状与手算例。先自己写，再逐层打开 [HINTS.md](HINTS.md)。`scaffolding.py` 提供模块初始化；通用attention、patch操作和EMA来自edu_core。AdaLN外围由本课给出，学生练习DDT两条输入路由与条件组装。

## 必须分清的约定

**时间**：`t=0`数据，`t=1`噪声；`x_t=(1-t)z+t*eps`，`target_velocity=eps-z`。采样用负步长。时间shift为 `a*t/(1+(a-1)*t)`、`a=sqrt(N*D/4096)`，训练和采样都使用它。采样先shift整条时间网格，再用相邻点的实际差值作Euler步长，不能shift输入时间却仍用均匀dt。

默认N=16、D=64，因此有效维度1024小于4096，a=.5，t=.5映射为1/3。该CPU缩尺会让shift偏向较低噪声；真实论文N=256、D=768时有效维度196608、a≈6.93，偏向较高噪声。保留论文参考维度，不偷偷将toy的n改小。

**噪声增强**：默认tau=.8，sigma=tau*abs(Normal(0,1))，每样本共享一个sigma；epsilon每个latent元素独立。只在stage1训练加噪声，stage2训练使用干净编码。允许显式normal/epsilon以便手算；sigma可超过tau，不能误写uniform。

**冻结**：encoder始终requires_grad=False/eval；stage1只有decoder训练，stage2整个RAE冻结/eval。EMA始终冻结/eval，在optimizer之后更新。顶层train()也维持这些约束。零输出和零gate初始化意味着首步部分上游梯度为0；demo运行多步后检查base、独立宽路径和条件映射真的更新，不要求随机loss单调下降。

**DDT**：base从x_t得到s；`condition=projector(SiLU(time[:,None]+s))`。宽head从另一个`noisy_embed(x_t)`开始，用condition生成每token的AdaLN shift/scale/gate。不是先把x_t压到32维再Linear到96，也不是简单concat两路。

## 来源与toy边界

|原始证据|本课保留|明确简化/差异|
|---|---|---|
|[论文§3、C.1–C.2](https://arxiv.org/html/2510.11690v1#S3)|冻结表示encoder、patch tokens、无affine LN、可训练ViT decoder|随机线性patch encoder；两层64维decoder；不加decoder CLS/位置编码；图像16×16|
|[论文§4.2–4.3](https://arxiv.org/html/2510.11690v1#S4.SS2)|N×D维度shift、半正态幅度增强|保留参考维度4096，CPU有效维度更小；未验证生成质量改善|
|[官方stage1/rae.py](https://github.com/bytetriper/RAE/blob/main/src/stage1/rae.py)|encode/decode边界与可逆token/grid|当前main.noising用tau*Uniform，课程明确遵循论文v1半正态；main可选数据集统计归一化不默认采用|
|[官方DDT.py](https://github.com/bytetriper/RAE/blob/main/src/stage2/models/DDT.py)|窄base、原噪声独立宽head、逐tokenAdaLN|base32/head96各2层；省RoPE/QKNorm/SwiGLU，使用LayerNorm/GELU和简单time MLP；无类别条件/CFG/AutoGuidance|
|[官方path.py](https://github.com/bytetriper/RAE/blob/main/src/stage2/transport/path.py)|data0/noise1方向、velocity target|均匀基础time与小步Euler；未实现原训练日程/分布式|

论文decoder使用L1+LPIPS+GAN及自适应权重；本课只训练L1，**没有LPIPS、对抗损失或判别器**。完整细节见 [recipe](recipe/README.md)。现有toy参数布局不兼容官方checkpoint，不静默转换。

可选诊断输出（默认不写文件）：

```bash
PYTHONPATH=general_utils/edu_core python3 -m mllm.rae.run_rae_demo --output-dir /tmp/rae-diagnostics
```

输出`losses.json`与input/reconstruction/generated三张PPM。显示映射固定[-1,1]并裁剪，仅用于观察数据流；短训练图像不能证明语义、分布拟合、论文速度提升或FID。真实数据、权重和评估路线见recipe。
