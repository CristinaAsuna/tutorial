# 教学设计与决策

## 表征辅助生成课程（2026-10-28）

按用户明确选择，在 mllm 下新增独立的 REPA 六关和 RAE 八关；生成课程的已有 DiT 学生文件不修改，两课不相互导入 reference。仅复用 edu_core 的 token attention、patchification、冻结与 EMA，不新增泛化框架。默认离线 CPU，不下载预训练编码器。

学习路线明确区分 foundations（基础算法前置）和 backbones（真实模型组件选修）。基础第 1、3 关后可直接进入 REPA 或 RAE，两课内部提供连续 flow、速度目标、负步长 Euler 与条件 Transformer 的微任务，不要求先完成生成模型目录。每个 helper 提供目的、轴推导、手算和三级提示；工程初始化与批量外围由 scaffolding 提供。

REPA 保留冻结干净 RGB 教师、独立生成 latent 路径、早期 hidden 的三层投影与 patch-wise 负余弦、联合 flow 训练和无教师采样。toy codec 不视作预训练 VAE，冻结随机教师不视作已训练语义编码器。RAE 保留无 affine 的逐 token LN、噪声增强解码、两个训练阶段、维度相关时间偏移及独立 noisy-latent 输入的宽 DDT head；窄主干提供逐 token AdaLN 条件，不用末尾 Linear 扩宽代替该路径。

两课统一 t=0 数据、t=1 噪声，训练目标 noise-clean，Euler 从 1 到 0；不沿用现有 flow_matching 的相反时间记号。RAE 以原始 v1 为合同，噪声强度为半正态，参考维度为 4096；当前官方代码的均匀噪声强度与新增统计归一化能力需在来源表中区分。默认 decoder 仅用 L1，论文 LPIPS/GAN、真实 DINO/SD-VAE 权重和 FID 协议留 recipe，均不声称已运行。

维护测试须拒绝常数假损失和错误目标、检查实际数值/非零梯度/参数更新，并在测试进程注入 TODO 检查 CLI，不要求学生文件始终未完成。零初始化层经过必要更新后再检查上游梯度。reference、替身 practice 接线、学生实际完成及真实数据指标分别记录。

## 目标

2026-10-08 确定：统一改造 ACT、DINO、Flamingo、I-JEPA、InstructBLIP、LLaVA、V-JEPA，采用细步骤与分层提示。验收目标是学习者独立完成 CPU 机制实现，真实数据与论文指标训练保留后续路线。

## 结构

- 每关先解释目的与数学/形状，再给小例子、编号 TODO、三级提示、局部检查、整模型接入。
- 默认 demo 运行完整参考解；practice 模式通过显式组合和覆盖方法运行学生代码，未完成时报告具体 TODO。
- 参考模式用于验证答案和验收工具；practice 模式用于验证填答。相同的形状不能证明相同的机制，检查必须覆盖数值、遮蔽、配对与梯度。
- 外围 toy 模型可提供，正在练习的机制必须学生实现。不要求从零重复写 tokenizer、数据读取和训练工程。

## 共享边界

已有 edu_core 是复用组件的归属，因此继续扩展，不新增另一套 utils。基础课程练一次，论文默认导入经过测试的组件。

无损 patchify 与可学习 PatchEmbed 是不同接口。共享 2D 局部像素顺序为 p,p,C，3D 为 t,p,p,C；网格宽度变化最快。attention 的 True 表示允许看见，但论文 target mask 的 True 表示选为目标，不能混用。

EMA helper 只更新 parameters；teacher frozen/eval、center buffers、momentum schedule 和调用顺序由论文策略明确管理。新增纯 self-attention block，避免纯 encoder 携带不用的 cross-attention 参数。已有参数布局不同的参考架构不强制迁移，保持已有 toy checkpoint 边界。

## 兼容与学习成果

MAE、BLIP-2 不做整套课程重写。已有学生填答必须保留，README 的未完成描述需结合实际代码理解。七篇原 lesson 的短签名不是稳定 API；缺失必要输入时明确更新，并同步接线和检查器。参考 demo 保持默认调用方式。

## 后续真实复现

先完成练习验收，再根据 recipe 明确原始论文版本、数据许可与格式、预训练权重、硬件预算、评估协议。toy 使用的精确补集、固定 crop 或简化多图可见性不自动成为论文原始合同。

## OpenVLA / Diffusion Policy 课程（2026-10-16）

新增原始 OpenVLA（2024）六关与 Diffusion Policy（RSS 2023）八关。Diffusion Policy 采用 CNN/FiLM 主线和完整 Transformer 扩展；两者共享轨迹、噪声日程、loss、EMA、采样和滚动控制。属于 embodied policy 教学，放在 mllm，论文专属 scheduler/动作 tokenizer 不提升为 edu_core。

初始化与批量外围单独提供于各篇 scaffolding.py；参考计算是独立函数，practice 显式构造所有 lesson 的 ops，并传入模型/网络，没有异常回退或运行时产品 monkeypatch。局部检查只替换选中关，提供其余参考外围以便独立练习；全 practice 始终使用全部学生机制。维护接线测试中的替身不写回 lesson，兼容后续学生填答。

OpenVLA 保留官方 256 边界/255 中心的离散行为、BOS 后插入视觉、动作与默认 EOS 监督、全词表生成、全参数梯度；toy 视觉与 LM 不兼容官方 checkpoint。Diffusion Policy 保留 square-cosine/epsilon 目标、EMA、观测缓存、DDPM/DDIM、To-1 执行起点和 Transformer 时间对齐 mask；单次随机 toy 不证明分布拟合或任务成功。两篇 recipe 均是外部官方实验路线与 metadata 骨架，没有隐藏的真实 trainer/rollout 实现。

## π0与可复用基础模型接口（2026-10-16）

新增原始π0七关：多相机/状态、prefix与suffix、三块mask/有效位置、逐层双流QKV、flow目标、训练更新、缓存Euler与重规划。统一采用openpi的t=1噪声、t=0动作、速度noise-actions与负Euler步长；避免将论文相反时间记号与代码混用。tiny骨干端到端训练，预训练SmolVLM测试冻结骨干、训练窄expert，分别记录证据。

共享edu_core.backbones按视觉、语言和分层K/V能力拆接口。初始化和 projected attention/RoPE 基础归共享包，π0自己的路由、损失、采样和训练留在课程。显式PrefixEncoding缓存不追加suffix，训练缓存保留梯度；tiny完整联合计算与缓存路径比较输出及梯度。参数初始化提供于scaffolding，student ops包含loss、训练步和采样，不回退参考答案。

用户选择先接入π0、LLaVA、OpenVLA，下载SmolVLM-256M。真实模型只由显式download命令获取，固定SHA记录在HF cache，运行local-only，默认离线验收不依赖transformers。loader尊重实际checkpoint的Idefics3架构并检查所有权重key，不以随机权重冒充预训练。LLaVA复用不重叠的vision/language子模块、保留projector/packing，使用有说明的patch pooling/合成CLS桥；OpenVLA只替换LM，并显式映射到256个非特殊动作ID，双视觉练习仍为toy。

全仓库审查见backbone_audit.md。BLIP系后续可适配raw vision/embedding forward，Flamingo还需逐层插入接口，视频JEPA需要独立视频能力；本次不为统一而覆盖旧课填答或改checkpoint布局。基础接口小课提供形状、mask、梯度、KV和实际模型的动手步骤。

## 新 session 的论文名称入口

2026-10-08 确定：在本仓库的新 session 中，仅输入论文名称默认请求生成或完善教学课程。根 AGENTS.md 指向 paper_workflow.md，携带学习者定位、提示深度、交付物和验证要求；不依赖前一 session 的聊天记录。只有论文身份存在实质歧义时再问，其他教学偏好沿用本仓库默认值。该工作流尚未通过独立新 session 的端到端行为验证。


## DINO家族递进设计（2026-10-17）

旧dino仍为DINO+iBOT入门；dinov2新增六关，dinov3为第4节Gram机制五关，不覆盖完整DINOv3架构。保留已有学生代码；本次只在空的DINO裁剪TODO中提供随机采样外围和新的确定性微任务。

通用patch/encoder/位置插值复用edu_core，家族heads、增广、目标和teacher策略留在论文目录。DINO共享block后参数布局改变，旧fused-qkv checkpoint不静默兼容。新课程使用根目录python -m入口；runner为子进程提供仓库根与共享包路径，不在产品代码修改sys.path。

机制验收覆盖内容关联、随机性、SK边际、不同mask数量的逐行归约、近邻排除自身、Gram不变性、梯度边界与快照时间轴。三层提示提供实际操作和伪代码，不把查看参考解作为唯一帮助。维护用替身接线与学生实际完成分开记录。

8步合成训练输出熵、特征方差与patch关系SVG，只用于诊断；不强制随机loss下降，不将短训练称为dense退化/修复复现。真实数据、完整架构和分布式路径在recipe说明为尚未实现的迁移工作。
