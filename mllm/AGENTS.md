# 课程质量约定

## 每篇四层

1. 参考解：完整、可运行、注明 toy 简化。
2. lesson：关键机制留给学生实现，外围工程和模块属性完整提供。
3. demo：默认参考模式；`--implementation practice` 实际运行学生实现。
4. recipe：真实数据、权重、训练条件与评估路线，不视作已完成论文复现。

## 每关必需内容

中文目标、前置知识、形状表或轴推导、可手算例子、编号小 TODO、常见错误、检查命令与整模型接入位置。每个大机制拆成能逐步检查的小任务；在 HINTS.md 提供思路、操作、伪代码三级提示。提示可选择阅读，不把“去看参考答案”作为唯一引导。

签名必须提供完成计算所需的模块、mask、labels 和维度信息。若已初始化模块，只让学生填写 forward 的关键计算；避免在同一 TODO 中同时设计完整网络和训练工程。

## 接线与验收

- `practice_<paper>.py` 显式组合前序 lesson。外围 backbone 可复用参考代码，但正在练习的机制不可转调答案。
- 用模块注入、组合或方法覆盖接线；禁止按异常回退答案，禁止运行时 monkeypatch 作为产品接线。
- 局部入口支持 `--lesson N --implementation reference|practice`，默认 practice。遇到 NotImplementedError 明确报告关卡；语义错误须失败，不能标记为未完成或跳过。
- reference/practice demo 使用相同机制验收。训练步骤也要调用相应实现。
- 检查内容顺序、信息遮蔽、梯度边界、参数更新和算法配对，不只检查 shape。
- 自动维护测试可用参考组件替换学生函数来验证组装，并用调用计数确认路由；这不代表 TODO 已完成。
- 基础只在 foundations 练一次，论文默认复用 edu_core；保留论文特有路由、损失与训练策略的练习。

## 教学边界

尊重当前课程定义：DINO+iBOT 机制 toy、原始 I-JEPA/V-JEPA、LLaVA-1.5、原始 Flamingo/InstructBLIP/ACT 核心机制。不要顺带换成新版算法。mask 不重叠、固定分辨率、toy tokenizer 或多图可见性策略等简化须明确说明；更接近论文的训练条件留在 recipe。
