# 教学设计与决策

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

## 新 session 的论文名称入口

2026-10-08 确定：在本仓库的新 session 中，仅输入论文名称默认请求生成或完善教学课程。根 AGENTS.md 指向 paper_workflow.md，携带学习者定位、提示深度、交付物和验证要求；不依赖前一 session 的聊天记录。只有论文身份存在实质歧义时再问，其他教学偏好沿用本仓库默认值。该工作流尚未通过独立新 session 的端到端行为验证。
