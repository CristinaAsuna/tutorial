"""第4关：不同样本遮挡数不同，应先按样本归约。
前置：DINO masked CE。student/targets B,N,K，patch_mask B,N bool。
手算：两行mask数1、3，masked CE和2、12 -> (2/1+12/3)/2=3，非14/4。
targets 已是 teacher 分配概率；本关不再次 softmax，不更新 center。
接入：model.losses；teacher 无 mask，student 有 mask，相同 crop 对应位置。
常见错误：全批 masked token 一起平均；分母只取被选中的样本数。
检查：python3 -m mllm.dinov2.check_lessons --lesson 4
"""

def ibot_loss(student, targets, patch_mask, student_temp=.1):
    # TODO 4.1: target detach，student/temperature log_softmax，类别轴CE -> B,N。
    # TODO 4.2: 每行mask CE和 / mask数量.clamp_min(1)。
    # TODO 4.3: 对全部B行 mean；没有mask的行贡献0，整批无mask也可backward。
    raise NotImplementedError('lesson4.ibot_loss TODO 4.1-4.3')
