# 当前验证状态

## 2026-10-28：REPA / RAE 教学闭环与回归

环境：macOS CPU，`/Users/max/codebase/.ml/.venv/bin/python`，Python 3.11.15、PyTorch 2.13.0、pytest 9.1.1。系统 `/usr/bin/python3` 没有 torch；下列命令使用已确认的环境实例，课程代码不硬编码该路径。所有命令从仓库根目录执行，无模型或数据下载。

|实际检查|结果|验收范围|
|---|---|---|
|REPA / RAE 专项 pytest|34 passed（13 + 21）|数值、错误假损失拒绝、冻结/更新、空间路由、宽 head、采样、替身接线、TODO/错误传播及诊断文件|
|两课统一 reference 入口|6 个入口通过|REPA 六关、RAE 八关；各自 demo 和维护接线测试|
|既有十二篇统一 reference 回归|36 个入口通过|局部检查、demo、接线；Diffusion Policy 含 CNN/Transformer。与新增两课合计十四篇、42 个入口，分批运行|
|共享 edu_core|22 passed|基础组件和 backbone 合同回归；本轮未改共享实现|
|REPA CPU demo|4 次训练更新及采样通过|最终 total=.68592727、velocity=.99958825、alignment=-.62732196；实际梯度/参数更新、冻结目标、EMA、无教师采样|
|RAE CPU demo|stage1、stage2 各 5 次更新及采样通过|最终 L1=.6596、flow=1.5737；encoder/decoder 阶段边界、窄 base/独立宽路径/条件映射更新、EMA 与 shifted Euler|
|原始 practice 第一关|两课均具体 TODO、exit 2|REPA clean_targets；RAE normalize_tokens；不是学生已完成填答|
|两课 recipe|metadata 校验通过|data_exists=false，RAE weights_exist=false；training_executed 和 paper_metrics_reproduced 都为 false|
|诊断与静态检查|通过|REPA JSON/SVG 解析；RAE JSON/三张 16×16 RGB 二进制 PPM 结构校验；compileall 与 git diff --check|

```bash
TUTORIAL_PY=/Users/max/codebase/.ml/.venv/bin/python
PYTHONPATH=.:general_utils/edu_core "$TUTORIAL_PY" -m pytest -q mllm/repa/tests mllm/rae/tests
PYTHONPATH=general_utils/edu_core "$TUTORIAL_PY" -m pytest -q general_utils/edu_core/tests
"$TUTORIAL_PY" mllm/check_tutorials.py --paper repa rae --implementation reference
"$TUTORIAL_PY" mllm/check_tutorials.py --paper dinov2 dinov3 act dino pi0 openvla diffusion_policy flamingo ijepa instructblip llava vjepa --implementation reference
PYTHONPATH=general_utils/edu_core "$TUTORIAL_PY" -m mllm.repa.run_repa_demo --output-dir /tmp/repa-final-diagnostics-20261028
PYTHONPATH=general_utils/edu_core "$TUTORIAL_PY" -m mllm.rae.run_rae_demo --output-dir /tmp/rae-final-diagnostics-20261028
"$TUTORIAL_PY" -m mllm.repa.recipe.validate_config
"$TUTORIAL_PY" -m mllm.rae.recipe.validate_config
"$TUTORIAL_PY" -m compileall -q mllm/repa mllm/rae mllm/check_tutorials.py
git diff --check
```

两课使用冻结随机 fixture 和合成图案，只证明机制与训练/采样路由。REPA codec 不是预训练 VAE，teacher 不是预训练 DINO；RAE stage1 只用 L1，没有论文 LPIPS/GAN。没有真实 ImageNet 训练、预训练权重接入、FID、论文加速或语义效果验证。参考与测试替身通过不代表学习者已填写练习。旧课程 MAE、BLIP-2、foundations 和预训练小模型未在本轮重跑，历史结果保留原日期。

## 2026-10-17：DINO系列实现与回归

本次使用macOS CPU、`/Users/max/codebase/.ml/.venv/bin/python`（Python 3.11、PyTorch 2.13.0），新增DINOv2六关和DINOv3 Gram五关，修正DINO温度、逐样本面积裁剪、共享encoder与语义检查。未运行真实数据或预训练模型测试；这些历史证据保留原日期。

|实际检查|结果|验收边界|
|---|---|---|
|全仓统一reference入口|36个入口通过|12篇局部检查、demo、接线；DP仍检查CNN/Transformer。参考通过不代表学生填答完成|
|DINO家族pytest|27 passed|多视角内容/随机性、mask像素隔离、非零center与温度、SK/iBOT/KoLeo、Gram梯度、快照刷新、checkpoint恢复、practice替身与CLI退出|
|共享edu_core pytest|22 passed|基础与已有backbone合同回归；本轮未改共享包实现|
|DINOv2/DINOv3结构化图案demo|各8步CPU训练通过|前者双head/SK/iBOT/KoLeo/EMA，后者早期快照/两阶段Gram/高分辨率对齐/刷新|
|诊断输出|两组JSON、SVG生成且结构解析通过|初始/最终patch关系；v3额外记录refinement前状态。未把热图视为dense指标|
|两课recipe默认配置|校验通过，data_exists=false|training_executed与paper_metrics_reproduced均false；只检验metadata|
|静态编译、git diff --check|通过|DINO三个目录与统一runner；无空白错误|

复验命令（仓库根；使用已安装torch的Python）：

```bash
TUTORIAL_PY=/Users/max/codebase/.ml/.venv/bin/python
"$TUTORIAL_PY" mllm/check_tutorials.py --implementation reference
PYTHONPATH=.:general_utils/edu_core "$TUTORIAL_PY" -m pytest -q mllm/dino/tests mllm/dinov2/tests mllm/dinov3/tests
PYTHONPATH=general_utils/edu_core "$TUTORIAL_PY" -m pytest -q general_utils/edu_core/tests
PYTHONPATH=general_utils/edu_core "$TUTORIAL_PY" -m mllm.dinov2.run_dinov2_demo --output-dir /tmp/dinov2-tutorial-diagnostics
PYTHONPATH=general_utils/edu_core "$TUTORIAL_PY" -m mllm.dinov3.run_dinov3_demo --output-dir /tmp/dinov3-tutorial-diagnostics
"$TUTORIAL_PY" -m mllm.dinov2.recipe.validate_config
"$TUTORIAL_PY" -m mllm.dinov3.recipe.validate_config
"$TUTORIAL_PY" -m compileall -q mllm/dino mllm/dinov2 mllm/dinov3 mllm/check_tutorials.py
git diff --check
```

原DINO仍保留legacy weight_norm弃用提示，但teacher采用重建加载而非deepcopy。共享SelfAttentionBlock替换了旧fused-qkv实现，参数布局变化已在课程说明，旧checkpoint不静默兼容。完整DINOv3架构、分布式训练、真实dense评估与论文指标没有在本次验证中执行。

## 2026-10-16历史验证

验证日期：2026-10-16。所有命令从仓库根执行；本次新增π0与共享基础模型接口。

环境：macOS CPU，`/Users/max/codebase/.ml/.venv/bin/python`，Python 3.11.15、PyTorch 2.13.0、pytest 9.1.1、transformers 5.15.0、huggingface_hub 1.27.0、safetensors 0.8.0、Pillow 12.3.0、torchvision 0.28.0。当前会话MPS不可用，测试采用CPU/float32/eager。普通课程依赖torch；预训练依赖单独声明于edu_core的pretrained extra。

## 离线验收

|范围|结果|证明的内容|
|---|---|---|
|共享组件与backbone合同|22 passed|原16项保持通过，新增lazy依赖、GQA手算/全遮蔽行、视觉投影梯度、mask/位置与可训练prefix K/V|
|十篇reference局部检查|50个关卡定义、58次关卡检查通过|原九篇43关，加π0七关；DP两种网络分别检查|
|十篇reference demo|11个模型配置通过|原十个配置加π0；有限loss、梯度与更新、适用的冻结/EMA/生成|
|十组接线验证|全部通过|学生函数调用计数，包含loss、训练与采样；替身不写回lesson|
|完整统一入口|30个入口全部通过|每篇局部检查、demo、接线各一组；不依赖预训练下载|
|π0特定语义|通过|相机/文字padding隔离、state不读actions、完整/缓存输出及梯度等价、padding梯度为0、缓存只读、每规划只建一次、新观测刷新缓存、Euler方向与Beta时间|
|π0未完成练习|具体TODO、非零退出|原始TODO及测试注入TODO可定位，不静默调用答案|
|π0 recipe|合法metadata通过、错误时间方向被拒绝|默认真实机器人数据路径不存在；不代表真实训练|
|静态编译和diff|通过|新增共享接口、π0、组件入口可编译，无diff空白错误|

共享RMSNorm使用torch基础算子，以避免对旧共享包下限引入nn.RMSNorm的隐含要求；另用本机PyTorch RMSNorm做数值对照。该兼容实现更改后，复验共享22项、π0全部3个统一入口和π0真实SmolVLM demo，通过；旧课程代码未再更改。

## 实际下载与预训练测试

已下载 `HuggingFaceTB/SmolVLM-256M-Instruct`，revision：

```text
7e3e67edbbed1bf9888184d9df282b700a323964
```

缓存位于 `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/`，约494MiB，包含精确revision的snapshot与tutorial-download.json。下载经联网执行完成；后续四个测试都从本地缓存加载，不再下载。checkpoint实际声明Idefics3，使用AutoModelForImageTextToText；missing/unexpected/mismatched权重检查通过。

|入口|实际结果|
|---|---|
|原生图文与接口|本地红方块问题输出 `The square is red.`；raw `(1,1024,768)`、projected `(1,64,576)`；30层K/V，首层 `(1,3,68,64)`；KV可对输入embedding反传，冻结模型参数无grad|
|π0 + SmolVLM|真实全部30层prefix K/V进入窄expert；expert有梯度/更新，骨干frozen/eval；10步Euler、重复性、观测编码一次、两轮重规划通过|
|LLaVA组件|真实视觉/语言组件，随机projector反传并更新；视觉/语言参数无grad；解除语言冻结不会解除视觉冻结|
|OpenVLA组件|动作ID避开所有special IDs、codec映射往返通过；冻结真实LM、两个toy视觉分支有梯度、projector更新、生成动作有限|

原生接口脚本本次一次墙钟记录约3.03秒，包含加载和多个特征/梯度检查，不是单次forward延迟或机器人控制频率。模型旧配置触发pad_token相关警告，但实际文本配置pad=2/vocab=49280，完整权重加载及运行通过。DINO仍有legacy weight_norm弃用提示。

## 复验命令

```bash
TUTORIAL_PY=/Users/max/codebase/.ml/.venv/bin/python
PYTHONPATH=general_utils/edu_core "$TUTORIAL_PY" -m pytest -q general_utils/edu_core/tests
"$TUTORIAL_PY" mllm/check_tutorials.py --implementation reference
PYTHONPATH=general_utils/edu_core "$TUTORIAL_PY" mllm/backbones/run_pretrained_demo.py
PYTHONPATH=general_utils/edu_core "$TUTORIAL_PY" mllm/pi0/run_pi0_demo.py --backbone smolvlm
PYTHONPATH=general_utils/edu_core "$TUTORIAL_PY" mllm/llava/run_backbone_demo.py
PYTHONPATH=general_utils/edu_core "$TUTORIAL_PY" mllm/openvla/run_backbone_demo.py
"$TUTORIAL_PY" mllm/pi0/recipe/validate_config.py
"$TUTORIAL_PY" -m compileall -q general_utils/edu_core mllm/pi0 mllm/backbones mllm/llava/backbone.py mllm/llava/run_backbone_demo.py mllm/openvla mllm/check_tutorials.py
git diff --check
```

更换环境使用对应Python，不在代码中硬编码本机解释器。普通入口不自动联网；显式重下固定版本：

```bash
PYTHONPATH=general_utils/edu_core python3 -m edu_core.backbones.download --revision 7e3e67edbbed1bf9888184d9df282b700a323964
```

## 保留证据与边界

基础六关reference、MAE/BLIP-2旧demo的最近独立验证仍为2026-10-08：当时通过，本次未重跑、不改其日期。上一轮2026-10-16的OpenVLA NumPy float32/float64边界oracle对照与两篇recipe检查仍保留为旧独立证据；本次重跑了其常规局部/整模型入口。

π0及新课程练习有意保留TODO；参考和测试替身成功不代表学生已独立完成填答。预训练接线没有绕过学生核心函数，仍可通过practice入口检查。所有课程原有lesson填答、用户AGENTS.md与paper_workflow.md修改保持不变。

本次没有真实机器人数据训练、官方π0/PaliGemma权重加载、完整256M基础模型端到端训练或论文指标复现。SmolVLM不是π0 checkpoint；LLaVA的projector和OpenVLA视觉双分支仍有明确toy差异。详细边界见各课README/recipe及backbone_audit.md。

更新规则：记录实际命令、环境、范围；未复验的旧结果保留日期，不把shape正确、配置通过或替身成功写成论文复现。
