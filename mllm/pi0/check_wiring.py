"""Maintenance-only oracle injection: proves routing, never fills student files."""
from collections import Counter
from contextlib import ExitStack
import importlib
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import torch
import reference_pi0 as reference
from lesson_registry import GROUPS
from run_pi0_demo import run_demo


def check_wiring():
    torch.set_num_threads(1)
    counts = Counter()
    originals = {}
    with ExitStack() as stack:
        for module_name, names in GROUPS:
            lesson = importlib.import_module(module_name)
            for name in names:
                original = getattr(lesson, name)
                originals[(module_name, name)] = original
                def routed(*args, _name=name, **kwargs):
                    counts[_name] += 1
                    return getattr(reference, _name)(*args, **kwargs)
                stack.enter_context(patch.object(lesson, name, routed))
        from practice_pi0 import build_toy_pi0 as builder
        for backbone in (None,):
            counts.clear()
            run_demo(builder)
            required = {name for _, names in GROUPS for name in names}
            missing = required - counts.keys()
            assert not missing, f"unrouted student functions: {missing}"
    # Ensure patches restore user functions; completed student answers remain supported.
    import ast
    import inspect
    for (module_name, name), original in originals.items():
        assert getattr(importlib.import_module(module_name), name) is original
        tree = ast.parse(inspect.getsource(original))
        body = tree.body[0].body
        if len(body) != 2 or not isinstance(body[-1], ast.Raise):
            continue  # learner has changed the initial stub; do not execute it with dummy inputs
        args = [None for p in inspect.signature(original).parameters.values()
                if p.default is inspect.Parameter.empty]
        try:
            original(*args)
        except NotImplementedError as error:
            assert name in str(error), (name, error)
    # A deliberate error in a core training function must escape the entire practice demo.
    module_name, names = GROUPS[5]
    lesson = importlib.import_module(module_name)
    class RoutingSentinel(RuntimeError):
        pass
    with ExitStack() as stack:
        for mod, entries in GROUPS:
            for name in entries:
                stack.enter_context(patch.object(importlib.import_module(mod), name, getattr(reference, name)))
        def poison(*args, **kwargs):
            raise RoutingSentinel("student training error must propagate")
        stack.enter_context(patch.object(lesson, "training_step", poison))
        try:
            run_demo(builder)
        except RoutingSentinel:
            pass
        else:
            raise AssertionError("practice swallowed a real error")
    # CLI negatives use explicit TODO substitutes, so the check still works after students finish.
    # Injection occurs only in this test subprocess; no lesson file is edited.
    folder = Path(__file__).resolve().parent
    code = """
import importlib, os, runpy, sys
os.chdir(sys.argv[1])
script = sys.argv[2]
from lesson_registry import GROUPS
for number, (module, names) in enumerate(GROUPS, 1):
    lesson = importlib.import_module(module)
    for name in names:
        def todo(*args, _message=f'lesson{number}.{name}: test TODO', **kwargs):
            raise NotImplementedError(_message)
        setattr(lesson, name, todo)
sys.argv = [script, '--implementation', 'practice']
runpy.run_path(script, run_name='__main__')
"""
    for script in ("check_lessons.py", "run_pi0_demo.py"):
        result = subprocess.run([sys.executable, "-c", code, str(folder), script],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 1, (script, result.returncode, result.stdout, result.stderr)
        assert "未完成" in result.stdout and "lesson" in result.stdout
    print("wiring PASS: every applicable lesson called; TODO/error propagation and CLI exit codes verified; exercises unchanged")


if __name__ == "__main__":
    check_wiring()
