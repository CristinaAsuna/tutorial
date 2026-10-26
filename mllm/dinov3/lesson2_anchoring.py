"""第2关：只固定局部关系，允许特征坐标系旋转。
前置：上一关Gram。输入student B,N,Ds、teacher B,N,Dt，可不同特征宽度。
手算：student Gram=I，teacher Gram全1，N=2时平方差sum=2、mean=.5。
公式Frobenius平方使用sum；官方MSE使用mean，权重需相应解释。
接入：refinement_loss -> 本关 -> 注入的lesson1.gram_matrix。
常见错误：直接MSE对齐features；teacher未detach；混淆sum/mean尺度。
检查：python3 -m mllm.dinov3.check_lessons --lesson 2
"""

def gram_loss(student, teacher, reduction='mean', remove_neg=False, *, gram_fn):
    # TODO 2.1: gram_fn(student) 与 gram_fn(teacher.detach())。
    # TODO 2.2: remove_neg为True时对两个Gram clamp_min(0)，对应官方选项。
    # TODO 2.3: 差平方，按reduction为mean/sum归约；保留student梯度。
    raise NotImplementedError('lesson2.gram_loss TODO 2.1-2.3')
