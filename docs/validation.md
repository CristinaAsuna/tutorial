# 当前验证状态

验证日期：2026-10-08。工作目录为仓库根目录。

环境：macOS CPU，`/Users/max/codebase/.ml/.venv/bin/python`，PyTorch `2.13.0`，pytest `9.1.1`。该环境尚未安装 edu_core，因此显式配置 PYTHONPATH；统一课程入口会为子进程配置该路径。

## 最近结果

|范围|结果|证明的内容|
|---|---|---|
|共享组件|16 passed|原组件、像素/tubelet顺序、逆变换、卷积投影对应、encoder残差|
|基础六关 reference|6 关通过|参考数值实现与基础检查器可运行|
|七篇 reference 局部检查|29 关通过|参考机制与局部检查合同可运行|
|七篇 reference demo|7 篇通过|CPU 前向、训练更新、冻结/EMA及适用的生成流程|
|七组接线验证|全部通过|测试替身下学生入口被实际调用；含 I-JEPA full-image teacher 回归与模式切换|
|MAE / BLIP-2 原 demo|均通过|既有参考运行兼容；两个目录没有文件修改|
|静态编译和 diff 检查|通过|Python 可编译；无 diff 空白错误|

统一入口共 21 个验收入口：七篇的局部检查、demo、接线验证各一组。接线验证包含 15 个测试用例和 ACT/Flamingo 两个测试替身完整 demo。

## 复验命令

```bash
TUTORIAL_PY=/Users/max/codebase/.ml/.venv/bin/python
PYTHONPATH=general_utils/edu_core "$TUTORIAL_PY" -m pytest -q general_utils/edu_core/tests
PYTHONPATH=general_utils/edu_core "$TUTORIAL_PY" mllm/foundations/check_lessons.py --implementation reference
"$TUTORIAL_PY" mllm/check_tutorials.py --implementation reference
"$TUTORIAL_PY" -m compileall -q general_utils/edu_core mllm
git diff --check
```

更换环境时使用对应 Python，并先安装 PyTorch/pytest；路径不是课程硬依赖。

## 练习状态与边界

新增练习刻意保留 TODO。practice 局部检查和 demo 已检查未完成错误定位，不会回退参考答案；未完成入口返回非零。基础全关检查会逐关报告未完成。测试替身仅在维护测试中使用，不会写入 lesson。

LLaVA/InstructBLIP 的大 packing 任务进一步拆成单样本 helper，批量循环、校验、去 padding 和补齐骨架已给出。JEPA 学生 loss 通过显式替换点进入 forward，不先运行参考损失再覆盖结果。

本次没有验证学生独立填答后的真实学习效果，也没有运行真实数据或论文规模训练。DINO 保留 legacy weight_norm，在当前 PyTorch 下有弃用提示，运行通过。真实复现条件与各篇 toy 简化见各目录 README/recipe。

更新规则：后续变更替换本文件的当前结果，并记录实际环境、命令和验证范围；不要将“参考通过”或“测试替身通过”升级为“练习完成”。
