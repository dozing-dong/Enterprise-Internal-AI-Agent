"""FastAPI 依赖注入函数。

所有路由通过 Depends(get_runtime) 获取运行时对象，
不直接引用模块级变量，便于测试时通过 dependency_overrides 替换。

组合根策略：
- `_runtime_factory` 决定如何构建 DemoRuntime，默认 = create_demo_runtime。
- `set_runtime_factory()` 允许在扩展场景下替换工厂。
- `reset_runtime()` 用于测试隔离，强制下次 init 重新构建。
"""

from typing import Callable

from backend.runtime import DemoRuntime, create_demo_runtime


RuntimeFactory = Callable[[], DemoRuntime]

_runtime: DemoRuntime | None = None
_runtime_factory: RuntimeFactory = create_demo_runtime


def set_runtime_factory(factory: RuntimeFactory) -> None:
    """替换运行时构建工厂，仅在 init_runtime 之前调用才会生效。"""
    global _runtime_factory
    _runtime_factory = factory


def init_runtime() -> None:
    """在应用 lifespan 启动阶段调用，预热并持有运行时对象。"""
    global _runtime
    _runtime = _runtime_factory()


def reset_runtime() -> None:
    """清空已注册的运行时对象，主要服务于测试隔离。"""
    global _runtime
    _runtime = None


def get_runtime() -> DemoRuntime:
    """FastAPI 依赖函数，返回已初始化的运行时对象。

    如果 lifespan 启动阶段没有成功初始化，会抛出 RuntimeError，
    触发全局异常处理器返回 503。
    """
    if _runtime is None:
        raise RuntimeError("运行时尚未初始化，请检查应用启动日志。")
    return _runtime
