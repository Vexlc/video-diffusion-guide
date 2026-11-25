"""用本地 HunyuanVideo 原源码复核实验 O 的时间表和欧拉步。无需模型权重。"""

import argparse
import importlib.util
from pathlib import Path

from flow_coupling import quantile, velocity
from hunyuan_trace import BUDGETS, RANKS, SHIFTS, knots, solve


def compare(root):
    import torch

    file = root/"hyvideo/diffusion/schedulers/scheduling_flow_match_discrete.py"
    spec = importlib.util.spec_from_file_location("original_hunyuan_scheduler",file)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    starts = [quantile(q,0) for q in RANKS]
    worst = 0.
    runs = 0
    for steps in BUDGETS:
        for shift in SHIFTS:
            scheduler = original.FlowMatchDiscreteScheduler(shift=shift,reverse=True,solver="euler")
            scheduler.set_timesteps(steps)
            expected = knots(steps,shift)
            torch.testing.assert_close(scheduler.sigmas,torch.tensor(expected),rtol=2e-6,atol=1e-6)
            torch.testing.assert_close(scheduler.timesteps,1000*torch.tensor(expected[:-1]),rtol=2e-6,atol=1e-4)
            paths = [solve(expected,z) for z in starts]
            state = torch.tensor(starts,dtype=torch.float32)
            for i,model_time in enumerate(scheduler.timesteps):
                tau = 1-float(scheduler.sigmas[i])
                output = torch.tensor([-velocity(value,tau) for value in state.tolist()])
                state, = scheduler.step(output,model_time,state,return_dict=False)
                truth = torch.tensor([path[i+1] for path in paths])
                torch.testing.assert_close(state,truth,rtol=2e-5,atol=2e-5)
                worst = max(worst,(state-truth).abs().max().item())
            runs += 1
    print(f"{runs} 组日程与逐步位置核对通过。最大位置差 {worst:.3g}。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo",type=Path,help="已检出的 HunyuanVideo 源码目录")
    compare(parser.parse_args().repo)
