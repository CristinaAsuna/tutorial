# I-JEPA 真实训练配方合同

此目录记录论文级训练需要的 ImageNet 数据、ViT-H/14、长训练、AMP/DDP、EMA schedule 与下游 linear probe 条件。它不是当前 CPU toy `reference_ijepa.py` 的扩容脚本，也不因配置存在而宣称复现论文指标。

运行 `python validate_config.py` 只检查配置合同；替换 `data_root` 并实现数据管线、checkpoint/resume、分布式统计和 probe 后，才可进入真实训练。
