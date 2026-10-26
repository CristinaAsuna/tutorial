# 从π0机制到真实机器人实验

本目录是外部官方训练与评估路线，配置检查不是本地真实trainer，也没有执行机器人rollout。

## 先区分两种真实权重

- 本仓库已提供SmolVLM-256M缓存路线：验证真实图文编码、K/V、expert更新。它不是π0 checkpoint，随机动作专家没有机器人能力，不能据此评估任务成功率。
- 原始π0实验使用PaliGemma和独立动作专家。进入真实实验时，独立checkout [官方openpi](https://github.com/Physical-Intelligence/openpi)，锁定commit和对应pi0模型配置；不能拿本toy的state_dict加载官方权重。

## 数据、动作与训练

从官方支持的ALOHA或DROID适配例子选择一条路线，使用其完整数据转换、坐标系、状态/动作单位、夹爪约定和normalization assets。metadata示例采用7维绝对joint target，但不同机器人维度不同；需要根据实际adapter一起修改state/action维度和mask，不能只改最后一层Linear。

每条episode保存有序多相机RGB、相机有效位、语言指令、state与action。按episode划分训练/验证，从训练split计算统计。明确当前时刻与动作目标对齐、horizon和执行步数、机器人控制频率；归一化不能抹掉joint与end-effector的控制语义。官方默认可能pad到更大的动作维度，按对应数据合同处理。

先运行官方单批forward/backward，核对mask、动作单位和冻结策略，再扩大真实微调；使用匹配checkpoint和assets，记录预训练/微调、LoRA/full、数据split、相机顺序、chunk长度、optimizer和随机种子。GPU依赖与显存要求以锁定版本的官方说明为准，本机CPU小模型耗时不能外推成真实控制频率。

本课程tiny默认端到端训练；SmolVLM demo默认冻结基础模型，以低成本验expert接线。后者不是原始π0训练配方，也不是π0.5的KI。若通过Python API将基础模型设为trainable，adapter不detach其KV，可传递flow loss；完整256M模型端到端训练的资源与效果没有在此承诺。

## 评估

先在held-out episode上检查物理动作误差与夹爪行为，再使用所选官方数据对应环境进行rollout，报告success rate、失败案例、种子、初始状态与checkpoint选择。视觉语言问答正确、flow loss有限或动作shape正确都不能替代机器人评估。

复现边界：真实PaliGemma权重、机器人数据、硬件/模拟环境、论文数据混合与指标尚未运行。`validate_config.py`只检查pi0身份、时间/速度方向、horizon和数据metadata；默认路径不存在会明确显示False。

```bash
python3 mllm/pi0/recipe/validate_config.py
```

来源：[π0原论文](https://arxiv.org/html/2410.24164v1)、[官方实现与模型](https://github.com/Physical-Intelligence/openpi)。
