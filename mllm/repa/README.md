# REPA：让生成模型的早期表征对齐干净视觉特征

本课对应 **Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think**（Yu 等，2024 / ICLR 2025），采用[论文 v3 §3.3 Eq.8–9、附录D](https://arxiv.org/html/2410.06940v3)。不是 REPA-E、REG 或其他后续方法。来源核对日期：2026-10-28。

REPA仍在生成latent空间去噪，只额外监督中间hidden；RAE则把表征编码器的输出作为生成空间。两课分别学习，不能把RAE latent直接叫作REPA原始配置。

## 从哪里开始

先做 [foundations](../foundations/README.md) 第1关像素切片、第3关attention、第6关EMA；理解broadcast、autograd与MSE即可进入本课。需要现成模型接口时再读[backbones](../backbones/README.md)，它不是本课的下载前置。

默认仅需PyTorch和edu_core，离线CPU，不下载权重。以下命令均从仓库根运行，使用已安装PyTorch的Python：

```bash
PYTHONPATH=general_utils/edu_core python3 -m mllm.repa.run_repa_demo
PYTHONPATH=general_utils/edu_core python3 -m mllm.repa.check_lessons --lesson 1
# 完成后换关卡编号；全部完成后运行自己的完整路径：
PYTHONPATH=general_utils/edu_core python3 -m mllm.repa.run_repa_demo --implementation practice
```

demo默认reference；检查默认practice。省略`--lesson`遍历六关，每关使用其余参考外围；完整practice全部调用学生机制。未完成明确报告函数TODO并退出2，真实数值错误直接失败。模块入口需`python3 -m`，不使用直接脚本路径或修改sys.path。

## 两条干净路径与一条带噪路径

```text
clean RGB (B,3,16,16) ──frozen toy codec──> z_clean (B,4,8,8)
        │                                    │ + noise/time
        └──frozen spatial teacher──> targets  z_t (B,4,8,8)
                                  (B,16,24)   │ patch2 -> (B,16,32)
                                        ↑     block1 hidden
                                3层MLP ←──────┤
                                              block2 -> velocity (B,4,8,8)
```

`B`是batch，`N=16`是4×4空间网格，`D=32`是student hidden，教师维为24。codec stride2再patch2，对应原图4×4像素；教师stride4也对应同一区域。宽度轴变化最快，不能随机打乱目标tokens。教师没有CLS。目标只读clean RGB，不读加噪latent；codec、teacher始终frozen/eval，无teacher EMA更新。

采用官方代码时间方向：

```text
t∈[0,1]，0=干净数据，1=噪声
z_t=(1-t)*z_clean+t*epsilon
velocity_target=epsilon-z_clean
L_velocity=mean((prediction-velocity_target)^2)
L_alignment=-mean_B,N(sum_D(normalize(projected)*normalize(detach(targets))))
L_total=L_velocity+0.5*L_alignment
```

负余弦可为负，不能以loss>0判断正确。采样从噪声`t=1`到数据`t=0`：`x_next=x-v/steps`，最后decode。外部teacher和projector仅训练使用，采样不得读取clean图像或执行alignment分支。

## 六关与文件职责

|关卡|亲手实现|手算/检查要点|
|---|---|---|
|1 干净目标|codec/teacher双路径，stop-gradient|空间(1,2)映射token6，目标无grad|
|2 flow|插值与速度目标|z=2,eps=0,t=.25 → 1.5,-2|
|3 hidden|adaLN-Zero、早层抽取、velocity重组|6路调制、1-based depth、project=False|
|4 alignment|逐patch负cos、联合loss|同向/正交/反向 → -1/0/+1|
|5 training|optimizer与EMA顺序|ema2/student6/m=.75 → 3|
|6 sampling|负步长Euler、codec decode|常速度-2，0 → 1 → 2|

先读lesson的形状和小例子，再填编号TODO；卡住逐层展开[HINTS](HINTS.md)。`scaffolding.py`提供模块初始化，`reference_repa.py`是独立参考解，`practice_repa.py`通过ops组合学生函数。共享attention、patchify/unpatchify、EMA无需再次设计。

`build_toy_repa() -> RepaSystem`。`system(images,labels,noise=None,time=None)`返回`loss/velocity_loss/alignment_loss`三个标量；`system.sample(labels,initial_noise,steps=8,use_ema=True)`返回`(B,3,16,16)`。reference/practice builder有相同返回接口。训练入口为`system.ops.training_step(system,optimizer,images,labels,noise=...,time=...)`，EMA在optimizer之后更新。

默认四步训练验收，避免adaLN与输出zero-init使第一步部分梯度为零而误判。检查四步后早/后attention、projector、patch/output参数实际非零梯度与更新；固定目标不更新。短训练不要求loss单调下降，不证明语义学习或生成质量。

```bash
PYTHONPATH=general_utils/edu_core python3 -m mllm.repa.check_lessons --implementation reference
PYTHONPATH=.:general_utils/edu_core python3 -m pytest -q mllm/repa/tests
PYTHONPATH=general_utils/edu_core python3 -m mllm.repa.run_repa_demo --output-dir /tmp/repa-diagnostics
python3 -m mllm.repa.recipe.validate_config
```

可选输出为loss分量JSON和输入/样本SVG；显示时将[-1,1]裁剪映射RGB，模型输出本身不裁剪。图像只用于检查数据流，不作为质量或加速证明。维护测试用替身计数接线，绝不写回lesson，也不代表学生已经完成。

## 原论文机制、官方代码与toy差异

|依据|保留内容|本课边界|
|---|---|---|
|[论文 Eq.8–9](https://arxiv.org/html/2410.06940v3#S3.SS3)|clean patch目标、hidden投影、负cos正则|单teacher，lambda=.5|
|[loss.py](https://github.com/sihyun-yu/REPA/blob/main/loss.py)|linear路径、noise-minus-clean、均值归约|固定噪声/时间可注入，uniform采样|
|[models/sit.py](https://github.com/sihyun-yu/REPA/blob/main/models/sit.py)|adaLN-Zero、早期hidden、三层SiLU projector|两层32维、第1层取hidden；坐标位置特征、简化时间MLP；原始B第4层/L与XL第8层|
|[train.py / utils.py](https://github.com/sihyun-yu/REPA/blob/main/train.py)|目标no_grad、patch空间对应、denoiser EMA|固定随机空间teacher与线性codec，无预训练语义、无VAE posterior|
|[samplers.py](https://github.com/sihyun-yu/REPA/blob/main/samplers.py)|时间1→0的Euler ODE可选路径|无CFG；论文指标主要用250步Euler-Maruyama SDE，本课未复现|

**随机teacher不是预训练DINOv2，固定codec不是预训练VAE。**本课只演示机制、形状与训练路由，无官方checkpoint兼容，无ImageNet/FID/17.5×加速验证。真实数据、DINOv2-B和SD-VAE路线见[recipe](recipe/README.md)。
