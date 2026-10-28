"""第2关：线性路径及速度监督。
输入clean/noise=(B,4,8,8)、time=(B,)，两输出保持latent形状。
公式z_t=(1-t)z+t eps，u=eps-z；0数据、1噪声。
手算z=2,eps=0,t=.25：z_t=1.5,u=-2；t=0/1得到两端点。
前置：广播；time需扩成(B,1,1,1)，不是(B,1)。
接入system.forward，固定noise/time允许独立检查与复现。
检查：python3 -m mllm.repa.check_lessons --lesson 2。
"""


def flow_pair(clean, noise, time):
    # TODO 2.1：检查匹配shape、time(B,)及[0,1]边界。
    # TODO 2.2：扩time广播，构造插值。
    # TODO 2.3：返回插值与noise-clean，不把目标写成clean-noise。
    raise NotImplementedError('REPA lesson2.flow_pair TODO 2.1-2.3')
