"""显式接线；不捕获TODO，不调用参考核心计算。"""
import importlib
from types import SimpleNamespace
from .lesson_registry import LESSONS
from .scaffolding import RAECourse


def practice_ops():
    functions = {}
    for module_name, names in LESSONS:
        module = importlib.import_module('.'+module_name, __package__)
        functions.update({name: getattr(module, name) for name in names})
    return SimpleNamespace(**functions)


def build_toy_rae():
    course = RAECourse(practice_ops())
    course.set_stage(1)
    return course
