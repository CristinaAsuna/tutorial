"""第2关：两个目标需要独立 head；teacher 必须从 student 同权重开始。
前置：nn.Module、state_dict、requires_grad。scaffolding.py 已初始化全部模块。
形状：CLS B,D / patch B,N,D，经各自 head -> B,K / B,N,K。
手算：student 参数=2、teacher=9，初始化后 teacher=2，且无梯度。
接入：build_toy_dinov2 -> scaffolding.build_modules -> 两个本关函数。
常见错误：独立模式误用同一个对象；只 eval 未冻结；冻结了 student。
检查：python3 -m mllm.dinov2.check_lessons --lesson 2
"""

def choose_patch_head(cls_head, candidate_patch_head, shared_head=False):
    # TODO 2.1: shared_head=True 返回 cls_head；否则返回已初始化 candidate。
    raise NotImplementedError('lesson2.choose_patch_head TODO 2.1')


def initialize_teacher(student, teacher):
    # TODO 2.2: teacher.load_state_dict(student.state_dict())。
    # TODO 2.3: 只冻结 teacher 参数，设置 eval；不返回替代模型。
    raise NotImplementedError('lesson2.initialize_teacher TODO 2.2-2.3')
