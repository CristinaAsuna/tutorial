"""关卡 2：生成时空 cuboid 与补集

前置：2D 矩形 mask 扩展。
形状合同：target/context bool [B,Tg*Hg*Wg]，默认2块。
手算例子：(Tg,Hg,Wg)=(2,2,2)，坐标(1,0,1) flatten index=5。
编号 TODO：1. 校验 block。2. 每样本放置 num_targets 块。3. 拒绝重叠。4. flatten/取补集。
常见错误：各样本 masked 数必须相同；不能满遮挡。
检查：python3 check_lessons.py --lesson 2 --implementation practice
提示：HINTS.md 第 2 关；参考检查可加 --implementation reference。
"""
import torch


def cuboid_mask(grid: tuple[int,int,int], start: tuple[int,int,int], block: tuple[int,int,int]) -> torch.Tensor:
    """微任务 1：固定 cuboid → bool[N]；(2,2,2) 单点(1,0,1) 只设 index5。"""
    # TODO 1a: 在 torch.zeros(grid,dtype=torch.bool) 上填时间/高/宽切片。
    # TODO 1b: flatten，time 最慢、width 最快。
    raise NotImplementedError("lesson2.cuboid_mask TODO 1")


def context_complement(target: torch.Tensor) -> torch.Tensor:
    """微任务 2：target bool[B,N] → context bool[B,N]，验证每行 target/context 非空。"""
    # TODO 2a: 检查 dtype、每行至少一个 True 与 False；2b: 返回 ~target。
    raise NotImplementedError("lesson2.context_complement TODO 2")


def make_masks(batch: int, grid: tuple[int, int, int], block: tuple[int, int, int], num_targets: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """有限回溯骨架给出；不会因随机早期放置产生偶然失败。"""
    import math
    if batch<=0 or num_targets<=0 or min(*grid,*block)<=0 or any(b>g for b,g in zip(block,grid)) or num_targets*math.prod(block)>=math.prod(grid):
        raise ValueError("blocks must fit and leave context tubelets")
    gt,gh,gw=grid;bt,bh,bw=block
    candidates=[cuboid_mask(grid,(t,h,w),block) for t in range(gt-bt+1) for h in range(gh-bh+1) for w in range(gw-bw+1)]
    rows=[]
    for _ in range(batch):
        order=torch.randperm(len(candidates)).tolist()
        def search(occupied,count,begin):
            if count==num_targets:return occupied
            for offset in range(begin,len(order)):
                mask=candidates[order[offset]]
                if (occupied&mask).any():continue
                answer=search(occupied|mask,count+1,offset+1)
                if answer is not None:return answer
            return None
        row=search(torch.zeros(math.prod(grid),dtype=torch.bool),0,0)
        if row is None:raise ValueError("cannot place requested non-overlapping cuboids")
        rows.append(row)
    target=torch.stack(rows)
    return target,context_complement(target)
