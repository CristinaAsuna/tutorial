"""关卡 1：矩形 target 与 context 补集

前置：网格索引/布尔集合。
形状合同：targets: list[bool N], context: bool N。
手算例子：2×3 网格左上 1×2 target=[T,T,F,F,F,F]，context 逐位取反。
编号 TODO：1. 校验网格。2. 枚举合法起点。3. 构造矩形并拒绝重叠。4. union 后取补集。
常见错误：必须有 context；采样要有限终止而非无限 while。
检查：python3 check_lessons.py --lesson 1 --implementation practice
提示：HINTS.md 第 1 关；参考检查可加 --implementation reference。
"""
import torch


def rectangle_mask(grid: tuple[int, int], start: tuple[int, int], block_size: tuple[int, int]) -> torch.Tensor:
    """微任务 1：固定起点，返回 bool [Gh*Gw]；(2,3),(0,0),(1,2) → TTFFFF。"""
    gh, gw = grid
    r, c = start
    bh, bw = block_size
    # TODO 1a: torch.zeros(gh,gw,dtype=torch.bool)。
    # TODO 1b: 将 [r:r+bh,c:c+bw] 设 True；1c: flatten，width 变化最快。
    raise NotImplementedError("lesson1.rectangle_mask TODO 1")


def context_complement(targets: list[torch.Tensor]) -> torch.Tensor:
    """微任务 2：targets K×bool[N] → context bool[N]，True 表示可见。"""
    # TODO 2a: stack 后 any(dim=0) 得所有 target 的 union。
    # TODO 2b: 按位 ~union；若没有可见 patch，抛 ValueError。
    raise NotImplementedError("lesson1.context_complement TODO 2")


def sample_block_masks(grid: tuple[int, int], num_targets: int, block_size: tuple[int, int]) -> tuple[list[torch.Tensor], torch.Tensor]:
    """搜索骨架已给出；学生只实现上面的索引、补集微任务。"""
    gh,gw=grid;bh,bw=block_size
    if min(gh,gw,bh,bw,num_targets)<=0 or bh>gh or bw>gw or num_targets*bh*bw>=gh*gw:
        raise ValueError("blocks must fit and leave context patches")
    candidates=[rectangle_mask(grid,(r,c),block_size) for r in range(gh-bh+1) for c in range(gw-bw+1)]
    order=torch.randperm(len(candidates)).tolist()
    def search(occupied, chosen, begin):
        if len(chosen)==num_targets:return chosen
        for offset in range(begin,len(order)):
            mask=candidates[order[offset]]
            if (occupied & mask).any():continue
            answer=search(occupied|mask,chosen+[mask],offset+1)
            if answer is not None:return answer
        return None
    targets=search(torch.zeros(gh*gw,dtype=torch.bool),[],0)
    if targets is None:raise ValueError("cannot place requested non-overlapping blocks")
    return targets,context_complement(targets)
