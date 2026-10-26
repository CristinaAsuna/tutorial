"""第1关：dense feature 表示每个patch，Gram表示patch之间的关系。
前置：DINOv2；矩阵乘法，L2 normalize。
形状：X B,N,D -> normalize(D) -> X @ X.T 为 B,N,N。
手算：两个正交patch [1,0],[0,2] 归一化后Gram为2x2单位矩阵。
接入：lesson2.gram_loss -> 本关；student分支保留梯度。
常见错误：构造DxD通道矩阵；沿N归一化；把CLS包含进patch网格。
检查：python3 -m mllm.dinov3.check_lessons --lesson 1
"""

def gram_matrix(features):
    # TODO 1.1: features.float()，沿最后D维做L2 normalize。
    # TODO 1.2: 与 transpose(-1,-2) 相乘，返回B,N,N。
    raise NotImplementedError('lesson1.gram_matrix TODO 1.1-1.2')
