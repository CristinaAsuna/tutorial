"""Run reference or practice lesson checks and CPU demos from the repo root."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


DEMOS = {"dinov2": "run_dinov2_demo.py", "dinov3": "run_dinov3_demo.py", "act": "run_act_demo.py", "dino": "run_dinov2_demo.py",
         "pi0": "run_pi0_demo.py",
         "openvla": "run_openvla_demo.py", "diffusion_policy": "run_diffusion_policy_demo.py",
         "flamingo": "run_flamingo_demo.py", "ijepa": "run_ijepa_demo.py",
         "instructblip": "run_instructblip_demo.py", "llava": "run_llava_demo.py",
         "vjepa": "run_vjepa_demo.py"}

WIRING = {"dinov2": ["-m", "pytest", "-q", "mllm/dinov2/tests"],
          "dinov3": ["-m", "pytest", "-q", "mllm/dinov3/tests"], "act": ["check_wiring.py"], "flamingo": ["check_wiring.py"],
          "pi0": ["check_wiring.py"],
          "openvla": ["check_wiring.py"], "diffusion_policy": ["check_wiring.py"],
          "llava": ["test_practice_wiring.py"], "instructblip": ["test_practice_wiring.py"],
          "dino": ["-m", "pytest", "-q", "tests"],
          "ijepa": ["-m", "pytest", "-q", "tests"],
          "vjepa": ["-m", "pytest", "-q", "tests"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", nargs="+", choices=DEMOS, default=list(DEMOS))
    parser.add_argument("--implementation", choices=("reference", "practice"), default="reference")
    parser.add_argument("--checks", choices=("lessons", "demo", "wiring", "all"), default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    core = root.parent / "general_utils" / "edu_core"
    env["PYTHONPATH"] = str(core) + os.pathsep + str(root.parent) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    results = []
    for paper in args.paper:
        scripts = []
        if args.checks in ("lessons", "all"):
            scripts.append("check_lessons.py")
        if args.checks in ("demo", "all"):
            scripts.append(DEMOS[paper])
        for script in scripts:
            print(f"\n[{paper}/{script}] {args.implementation}", flush=True)
            extra = ["--backbone", "all"] if paper == "diffusion_policy" else []
            packaged = paper in ("dinov2", "dinov3")
            command = ["-m", f"mllm.{paper}.{Path(script).stem}"] if packaged else [script]
            result = subprocess.run([sys.executable, *command, "--implementation", args.implementation, *extra],
                                    cwd=root.parent if packaged else root / paper, env=env, check=False)
            results.append((paper, script, result.returncode))
        if args.checks in ("wiring", "all"):
            print(f"\n[{paper}/wiring] 测试替身验证接线，不完成练习", flush=True)
            result = subprocess.run([sys.executable, *WIRING[paper]],
                                    cwd=root.parent if paper in ("dinov2", "dinov3") else root / paper,
                                    env=env, check=False)
            results.append((paper, "wiring", result.returncode))
    failures = [item for item in results if item[2]]
    if failures:
        print("\n未通过的入口：", flush=True)
        for paper, script, code in failures:
            print(f"  {paper}/{script}: exit {code}")
        return 1
    print(f"\n全部通过：{len(results)} 个验收入口（课程模式 {args.implementation}；接线测试独立验证）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
