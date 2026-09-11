# DINOv2/iBOT recipe track

该轨道要求真实 multi-crop augmentation、teacher EMA momentum/temperature/weight-decay schedules、跨卡 center 同步、patch masking 和 AMP/FSDP 或 DDP。公开代码可以复现 DINO/iBOT 的训练机制；DINOv2 的专有数据筛选、规模与基础设施不可由本目录宣称严格复现。

使用该配置时，必须分别记录数据筛选版本、global/local crop 策略、teacher momentum 曲线、跨卡 world size，以及 k-NN/linear probe 评估。顶层 toy 中的 shared prototype head 是教学简化，不是论文规模配置。
